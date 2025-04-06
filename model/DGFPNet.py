import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm
import math
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set


def cos_similar(q, s, flag):
    # flag:-1 0 1
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    cosine_eps = 1e-7
    tmp_query = q  # [1,2048,60,60]
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)  #
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  #
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)  #
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)  #
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)

    if flag == 1:
        return similarity
    else:
        if flag == -1:
            similarity = similarity.max(1)[0]
        similarity = similarity.view(bsize, sp_sz * sp_sz)  # [1,1,3600]
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)  # resize [1,1,60,60]
        return corr_query

def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        print(args.ppm_scales)  #
        # assert classes > 1  #
        from torch.nn import BatchNorm2d as BatchNorm
        self.ppm_scales = args.ppm_scales  #
        models.BatchNorm = BatchNorm
        self.arr_100 = torch.arange(0, 1, 0.01, dtype=torch.float32).cuda()
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.use_coco = args.use_split_coco
        self.val_size = args.val_size
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq / 2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        assert self.layers in [50, 101, 152]
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = '/mnt/data2/sophia_huang_data/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split,
                                                                                  backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        # Gram and Meta
        # self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        # self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # meta leaner
        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )

        self.up_query = nn.Sequential(
            nn.Conv2d(1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.inter_supp = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim,kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduce_dim, reduce_dim,kernel_size=1, padding=0, bias=False),
            # nn.Dropout(p=drop_rate),
            nn.Sigmoid())
        self.inter_query = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.pyramid_bins = args.ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )
        self.pool1_3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool3_1 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(reduce_dim * 4 + 1, reduce_dim, (1, 3), 1, (0, 1), bias=False),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 4 + 1, reduce_dim, (3, 1), 1, (1, 0), bias=False),
            nn.ReLU(inplace=True))
        self.conv13_31 = nn.Sequential(
            nn.Conv2d(reduce_dim * 3, reduce_dim, 3, 1, 1, bias=False),
            nn.ReLU(True))
        self.conv13_31_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 4 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)
        # 加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。
        # 类似list
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(reduce_dim + 1, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        self.pseudo_down_fore = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pseudo_down_back = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pseudo_cls_fore = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.pseudo_cls_back = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        # self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        # self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))
        self.low_fea_id = args.low_fea[-1]
        # all fre
        if self.use_coco & (self.val_size == 641):
            if self.vgg:
                c_wh_arr = ((64, 321), (128, 161), (256, 41), (512, 41), (512, 41))
            else:
                c_wh_arr = ((128, 161), (256, 161), (512, 81), (1024, 81), (2048, 81))
        else:
            if self.vgg:
                c_wh_arr = ((64, 237), (128, 119), (256, 30), (512, 30), (512, 30))
            else:
                c_wh_arr = ((128, 119), (256, 119), (512, 60), (1024, 60), (2048, 60))
        mapper_x = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]
        mapper_y = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
                    0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
        mapper_x = [temp_x * (c_wh_arr[3][1] // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (c_wh_arr[3][1] // 7) for temp_y in mapper_y]

        self.weight = self.get_dct_filter(c_wh_arr[3][1], c_wh_arr[3][1], mapper_x, mapper_y, c_wh_arr[3][0])
        self.weight2 = self.get_dct_filter(c_wh_arr[3][1], c_wh_arr[3][1], mapper_x, mapper_y, reduce_dim)
        self.freeze_bn()
        self.flag = 0
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': model.cls.parameters()},
                {'params': model.down_query.parameters()},
                {'params': model.up_query.parameters()},
                {'params': model.inter_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.inter_supp.parameters()},
                {'params': model.conv1_3.parameters()},
                {'params': model.conv3_1.parameters()},
                {'params': model.conv13_31.parameters()},
                {'params': model.conv13_31_cls.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.beta_conv.parameters()},
                {'params': model.inner_cls.parameters()},
                {'params': model.res1.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.res3.parameters()},
                {'params': model.alpha_conv.parameters()},
                {'params': model.pseudo_down_fore.parameters()},
                {'params': model.pseudo_down_back.parameters()},
                {'params': model.pseudo_cls_fore.parameters()},
                {'params': model.pseudo_cls_back.parameters()},
                # {'params': model.cls_merge.parameters()},
                # {'params': model.gram_merge.parameters()},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        # torch.cuda.synchronize()
        # end = time.time()

        cosine_eps = 1e-7
        x_size = x.size()
        bs = x_size[0]
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0  #
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        #   Query Feature
        with torch.no_grad():  #
            query_feat_0 = self.layer0(x)  #
            query_feat_1 = self.layer1(query_feat_0)  #
            query_feat_2 = self.layer2(query_feat_1)  #
            query_feat_3 = self.layer3(query_feat_2)  #
            query_feat_4 = self.layer4(query_feat_3)  #
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)
        base_out = self.learner_base(query_feat_4)
        base_out_soft = base_out.softmax(1)
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                # c_mask_1 = (c_id_array != c_id)
                # base_mask = base_out_soft[b_id, c_mask_1, :, :].unsqueeze(0)
                # base_mask = base_mask.max(1)[1].unsqueeze(1)
                # base_mask[base_mask > 0] = 1
            base_map = torch.cat(base_map_list, 0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)


        #   Support Feature
        mask_list = []
        corr_query_mask4_list = []
        corr_query_mask3_list = []
        supp_feat_list = []
        supp_feat_main_list = []
        supp_feat_comple_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
            # que4_mask
            resize_size = supp_feat_4.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                     align_corners=True)
            tmp_supp_feat_4 = supp_feat_4 * tmp_mask
            area_4 = (torch.sum(tmp_mask, dim=[2, 3]) + 0.0005).unsqueeze(2).unsqueeze(2)
            sup_pro_feat_4 = (torch.sum(tmp_supp_feat_4, dim=[2, 3])).unsqueeze(2).unsqueeze(2)
            sup_pro_4 = sup_pro_feat_4 / area_4
            s_org = query_feat_4
            s = sup_pro_4
            corr_que_4 = cos_similar(s_org, s, 0)
            corr_query_mask4_list.append(corr_que_4.unsqueeze(1))

            # single feature3
            bsize, ch_sz, sp_sz, _ = supp_feat_3.size()[:]
            temp_supp_feat_3 = supp_feat_3.clone()
            tmp_fore_mask = F.interpolate(mask_list[i], size=(sp_sz, sp_sz), mode='bilinear',
                                          align_corners=True)
            area = (torch.sum(tmp_fore_mask, dim=[2, 3]) + 0.0005).unsqueeze(2).unsqueeze(2)
            temp_supp_feat_3 = temp_supp_feat_3 * tmp_fore_mask  # 4,1024,60,60

            #  supp_feat_3_prototype MFP
            supp_feat_3_proto_arr = self.get_fre_pro(temp_supp_feat_3, self.weight, area)


            # DGSM
            # supp_feat_3_mask
            corr_sup3_mask_list = self.get_fre_mask(supp_feat_3, supp_feat_3_proto_arr)
            # supp_brach similar index
            dis_arr1 = self.get_fre_sim_index(corr_sup3_mask_list, tmp_fore_mask, sp_sz)
            _, supp_feat_3_proto_index1 = dis_arr1.topk(10, dim=1, largest=False)
            supp_feat_3_proto_index1, _ = supp_feat_3_proto_index1.sort(dim=1)
            # que_feat_3_mask
            corr_que3_mask_list = self.get_fre_mask(query_feat_3, supp_feat_3_proto_arr)
            que_fore_mask = corr_que_4 * corr_que3_mask_list[:, 0, :, :, :] * (1.0 - base_map)
            dis_arr2 = self.get_fre_sim_index(corr_que3_mask_list, que_fore_mask, sp_sz)
            _, supp_feat_3_proto_index2 = dis_arr2.topk(10, dim=1, largest=False)
            supp_feat_3_proto_index2, _ = supp_feat_3_proto_index2.sort(dim=1)
            supp_feat_3_proto_index_all = []
            for j in range(bsize):
                supp_feat_3_proto_index_temp = torch.cat([supp_feat_3_proto_index1[j], supp_feat_3_proto_index2[j]],
                                                         dim=0)
                supp_feat_3_proto_index_temp = supp_feat_3_proto_index_temp.unique(dim=0)
                supp_feat_3_proto_index_all.append(supp_feat_3_proto_index_temp)
            # DGSM_final_selected_mask
            predict_que_mask_all = []
            for j in range(bsize):
                predict_que_mask = corr_que3_mask_list[j, supp_feat_3_proto_index_all[j], :, :, :]
                predict_que_mask = \
                    predict_que_mask.contiguous().view(1, len(supp_feat_3_proto_index_all[j]), sp_sz * sp_sz).max(
                        1)[0]  # [4,3600]
                predict_que_mask = (predict_que_mask - predict_que_mask.min(1)[0].unsqueeze(1)) / (
                        predict_que_mask.max(1)[0].unsqueeze(1) - predict_que_mask.min(1)[0].unsqueeze(1) + cosine_eps)
                predict_que_mask = predict_que_mask.contiguous().view(1, 1, sp_sz, sp_sz)
                predict_que_mask_all.append(predict_que_mask)
            predict_que_mask_all = torch.cat(predict_que_mask_all, dim=0)
            corr_query_mask3_list.append(predict_que_mask_all.unsqueeze(1))  # [1,4,1,60,60]

            # FGM_support_feature
            supp_feat_temp = torch.cat([supp_feat_3, supp_feat_2], 1)  # torch.Size([4, 1536, 60, 60])
            supp_feat_temp = self.down_supp(supp_feat_temp)  # 减少通道。中水平特征[4, 256, 60, 60]
            supp_feat_inter_temp = supp_feat_temp.clone()
            supp_feat_temp = supp_feat_temp * tmp_fore_mask
            supp_feat_inter = self.inter_supp(supp_feat_temp)
            tmp_supp_feat_single_proto = torch.sum(supp_feat_temp, dim=[2, 3]).unsqueeze(2).unsqueeze(2) / area

            # DGSM_final_main_prototype
            supp_feat_main_list.append(tmp_supp_feat_single_proto.unsqueeze(1))

            # DGSM_final_comple_prototype
            que_feat_3_comple_proto_arr = self.get_comple_prototype(self, bsize, supp_feat_3_proto_index_all, supp_feat_temp, area)
            supp_feat_comple_list.append(que_feat_3_comple_proto_arr.unsqueeze(1))
            supp_feat_list.append((supp_feat_inter * supp_feat_inter_temp).unsqueeze(1))
        # final_mask
        if self.shot > 1:
            # [B,S,C,60,60]
            corr_query_mask3 = torch.cat(corr_query_mask3_list, 1)
            s_b, _, _, s_w, s_h = corr_query_mask3.size()[:]
            corr_query_mask3 = corr_query_mask3.max(1)[0]  # [B,S,60,60]
            corr_query_mask3 = corr_query_mask3.contiguous().view(s_b, s_w * s_h)  # [1,60*60]
            corr_query_mask3 = (corr_query_mask3 - corr_query_mask3.min(1)[0].unsqueeze(1)) / (
                    corr_query_mask3.max(1)[0].unsqueeze(1) - corr_query_mask3.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_mask3 = corr_query_mask3.contiguous().view(s_b, 1, s_w, s_h)
            corr_query_mask4 = torch.cat(corr_query_mask4_list, 1).mean(1)
        else:
            corr_query_mask3 = torch.cat(corr_query_mask3_list, 1)[:,0,:,:,:]
            corr_query_mask4 = torch.cat(corr_query_mask4_list, 1)[:,0,:,:,:]
        corr_attention_mask = corr_query_mask3 * corr_query_mask4 * (1.0-base_map)

        # final_supp_feat
        supp_feat = torch.cat(supp_feat_list, 1).mean(1)
        # FGM_que_feature
        query_feat_mid_1 = torch.cat([query_feat_3, query_feat_2], 1)  # [1,1536,60,60]
        query_feat_mid_1 = self.down_query(query_feat_mid_1)  # [1,256,60,60]
        query_feat_mid_2 = self.up_query(corr_query_mask3 * corr_query_mask4)
        # final_query_feat
        query_feat = self.inter_query(torch.cat([query_feat_mid_1, query_feat_mid_2], dim=1))
        query_feat_org = query_feat.clone()
        # final_prototype
        supp_feat_main = torch.cat(supp_feat_main_list, 1).mean(1)
        supp_feat_comple = torch.cat(supp_feat_comple_list, 1).mean(1)

        # MDFEDM_feature_enrichment_part
        query_feat, out_list = self.feature_enrichment_part(query_feat, supp_feat, supp_feat_main,
                                                            supp_feat_comple, corr_attention_mask)
        # MDFEDM_segmentation_part
        corr_attention_org = corr_query_mask3 * corr_query_mask4 * (1.0-base_map)
        pseudo_feat_mask_fore,pseudo_feat_fore = self.segment_part_pseudo_mask(corr_attention_org, corr_attention_mask,
                                                              query_feat_org, supp_feat_main, 1)
        pseudo_feat_mask_back,pseudo_feat_back = self.segment_part_pseudo_mask(corr_attention_org, corr_attention_mask,
                                                              query_feat_org, supp_feat_main, 0)
        pseudo_feat_mask = pseudo_feat_mask_fore * corr_attention_org + pseudo_feat_mask_back * corr_attention_org

        query_feat = self.res3(torch.cat([query_feat, pseudo_feat_mask], 1)) + query_feat  # res2连个3*3卷积
        meta_out = self.cls(query_feat)
        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            pseudo_feat_fore = F.interpolate(pseudo_feat_fore, size=(h, w), mode='bilinear', align_corners=True)
            pseudo_feat_back = F.interpolate(pseudo_feat_back, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            tmp_que_fore_mask = (y_m == 1).float().unsqueeze(1)
            corr_attention_mask = F.interpolate(corr_attention_mask, size=(h, w), mode='bilinear',
                                                align_corners=True)
            corr_attention_mask_fore = corr_attention_mask.clone()
            corr_attention_mask_fore[corr_attention_mask < 0.35] = 0
            corr_attention_mask_fore[corr_attention_mask >= 0.35] = 1
            pseudo_query_ground_mask_fore = corr_attention_mask_fore * tmp_que_fore_mask
            pseudo_query_ground_mask_fore[corr_attention_mask_fore == 0] = 255

            corr_attention_mask_back = corr_attention_mask.clone()
            corr_attention_mask_back[corr_attention_mask < 0.35] = 1
            corr_attention_mask_back[corr_attention_mask >= 0.35] = 0
            pseudo_query_ground_mask_back = corr_attention_mask_back * tmp_que_fore_mask
            pseudo_query_ground_mask_back[corr_attention_mask_back == 0] = 255

            pseudo_loss = self.criterion(pseudo_feat_fore, pseudo_query_ground_mask_fore.squeeze(1).long()) \
                          + self.criterion(pseudo_feat_back, pseudo_query_ground_mask_back.squeeze(1).long())

            main_loss = self.criterion(meta_out, y_m.long())
            aux_loss = torch.zeros_like(main_loss).cuda()
            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y_m.long())
            aux_loss = aux_loss / len(out_list)
            return meta_out.max(1)[1], main_loss, aux_loss, pseudo_loss
        else:
            return meta_out, meta_out, base_out
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(len(mapper_x), channel, tile_size_x, tile_size_y).cuda()
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):  #
            for t_x in range(tile_size_x):  # H
                for t_y in range(tile_size_y):  # W
                    dct_filter[i, :, t_x, t_y] = (self.build_filter(t_x, u_x, tile_size_x)
                                                  * self.build_filter(t_y, v_y,tile_size_y))
        return dct_filter
    def get_fre_pro(self,temp_supp_feat,weight,area):
        supp_feat_proto_arr = []
        for j in range(49):
            # 获得DCT原型
            tmp_supp_feat_single = temp_supp_feat * (weight[j])
            tmp_supp_feat_single = torch.sum(tmp_supp_feat_single, dim=[2, 3])  # 4,1024
            tmp_supp_feat_single = tmp_supp_feat_single.unsqueeze(2).unsqueeze(2)
            tmp_supp_feat_single = tmp_supp_feat_single / area  # `4,1024,1,1
            supp_feat_proto_arr.append(tmp_supp_feat_single.unsqueeze(1))
        supp_feat_proto_arr = torch.cat(supp_feat_proto_arr, dim=1)  # [b,49,1024,1,1]
        return supp_feat_proto_arr

    def get_fre_mask(self, supp_feat, supp_feat_proto_arr):
        corr_sup_mask_list = []
        for j in range(49):
            s_org = supp_feat  # 查询集高水平特征
            s = supp_feat_proto_arr[:, j, :, :, :]
            corr_que = cos_similar(s_org, s, 0)
            corr_sup_mask_list.append(corr_que.unsqueeze(1))
        corr_sup_mask_list = torch.cat(corr_sup_mask_list, dim=1)
        return corr_sup_mask_list

    def get_fre_sim_index(self,corr_sup_mask_list,mask,sp_sz):
        dis_arr = []
        for j in range(49):
            rep_a = corr_sup_mask_list[:, j, :, :, :].contiguous().view(-1, sp_sz * sp_sz)
            rep_b = mask.contiguous().view(-1, sp_sz * sp_sz)
            distance = F.pairwise_distance(rep_a, rep_b, p=2)
            dis_arr.append(distance.unsqueeze(1))
        dis_arr = torch.cat(dis_arr, dim=1)
        return dis_arr

    def get_comple_prototype(self,bsize,supp_feat_3_proto_index_all,supp_feat_temp,area):
        que_feat_3_comple_proto_arr = []
        for j in range(bsize):
            # 获得DCT原型
            if supp_feat_3_proto_index_all[j][0] == 0:
                supp_feat_3_proto_index_all[j] = supp_feat_3_proto_index_all[j][1:]
            tmp_supp_feat_3_single = supp_feat_temp[j]
            tmp_supp_feat_3_single = tmp_supp_feat_3_single.unsqueeze(0).repeat(
                len(supp_feat_3_proto_index_all[j]),
                1, 1, 1)
            tmp_supp_feat_3_single = tmp_supp_feat_3_single * (self.weight2[supp_feat_3_proto_index_all[j]])
            tmp_supp_feat_3_single = tmp_supp_feat_3_single.mean(0).unsqueeze(0)
            tmp_supp_feat_3_single = torch.sum(tmp_supp_feat_3_single, dim=[2, 3])  # 1,256
            tmp_supp_feat_3_single = tmp_supp_feat_3_single.unsqueeze(2).unsqueeze(2)
            tmp_supp_feat_3_single = tmp_supp_feat_3_single / area[j]  # `1,256,1,1
            que_feat_3_comple_proto_arr.append(tmp_supp_feat_3_single)
        que_feat_3_comple_proto_arr = torch.cat(que_feat_3_comple_proto_arr, dim=0)  # [1,b,256,1,1]
        return que_feat_3_comple_proto_arr

    def segment_part_pseudo_mask(self, corr_attention_org, corr_attention_mask, query_feat_org, supp_feat_main,FB_flag):
        corr_query_mask = corr_attention_org
        if FB_flag == 1:# fore
            corr_query_mask[corr_attention_mask < 0.35] = 0
            corr_query_mask[corr_attention_mask >= 0.35] = 1
        else:# back
            corr_query_mask[corr_attention_mask < 0.35] = 1
            corr_query_mask[corr_attention_mask >= 0.35] = 0
        _, _, h_size, w_size = corr_query_mask.size()[:]
        supp_feat_exp = supp_feat_main.contiguous().expand(-1, -1, h_size, w_size)
        supp_feat_exp = supp_feat_exp * corr_query_mask
        pseudo_query_mask_feat = corr_query_mask * query_feat_org
        pseudo_feat = torch.cat([pseudo_query_mask_feat, supp_feat_exp], 1)  # bsize,512,60,60
        if FB_flag == 1:  # fore
            pseudo_feat = self.pseudo_down_fore(pseudo_feat)
            pseudo_feat = self.pseudo_cls_fore(pseudo_feat)
        else:  # back
            pseudo_feat = self.pseudo_down_back(pseudo_feat)
            pseudo_feat = self.pseudo_cls_back(pseudo_feat)
        pseudo_feat_mask = pseudo_feat.max(1)[1].unsqueeze(1)
        return pseudo_feat_mask, pseudo_feat
    def feature_enrichment_part(self,query_feat,supp_feat,supp_feat_main,supp_feat_comple,corr_attention_mask):
        out_list = []
        pyramid_feat_list = []
        feat_13_31=None
        for idx, tmp_bin in enumerate(self.pyramid_bins):  # pyramid_bins=[60, 30, 15, 8]
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
                supp_feat_bin = nn.AdaptiveAvgPool2d(bin)(supp_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
                supp_feat_bin = self.avgpool_list[idx](supp_feat)
            supp_feat_main_bin = supp_feat_main.contiguous().expand(-1, -1, bin, bin)
            supp_feat_comple_bin = supp_feat_comple.contiguous().expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_attention_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, supp_feat_main_bin, supp_feat_comple_bin,corr_mask_bin],
                                       1)  #
            if idx==0:
                feat_13_31=merge_feat_bin.clone()
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)  #

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()  #
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear',
                                             align_corners=True)  #
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)  #
                merge_feat_bin = self.alpha_conv[idx - 1](
                    rec_feat_bin) + merge_feat_bin  #

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin  #
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)  #
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)  #
            out_list.append(inner_out_bin)  #

        feat_13 = self.pool1_3(feat_13_31)
        feat_13 = self.conv1_3(feat_13).contiguous().expand(-1, -1, query_feat.size(2), -1)
        feat_31 = self.pool3_1(feat_13_31)
        feat_31 = self.conv3_1(feat_31).contiguous().expand(-1, -1, -1, query_feat.size(3))

        final_feat_13_31 = self.conv13_31(torch.cat([feat_13,feat_31, pyramid_feat_list[0]], 1))
        final_feat_13_31_cls = self.conv13_31_cls(final_feat_13_31)
        out_list.append(final_feat_13_31_cls)
        query_feat = torch.cat(pyramid_feat_list, 1)  #
        query_feat = self.res1(query_feat)  #
        query_feat = self.res2(torch.cat([query_feat,final_feat_13_31], 1))
        return query_feat, out_list
