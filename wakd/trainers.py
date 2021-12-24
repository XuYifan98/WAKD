from __future__ import print_function, absolute_import
import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from .utils.meters import AverageMeter
import time

class Trainer(object):
    #############################
    # Training module for
    # 1. "NetVLAD: CNN architecture for weakly supervised place recognition" (CVPR'16), loss_type='triplet'
    # 2. "Stochastic Attraction-Repulsion Embedding for Large Scale Localization" (ICCV'19), loss_type='sare_ind' or 'sare_joint'
    #############################
    def __init__(self, model, margin=0.3, gpu=None, temp=0.07):
        super(Trainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.margin = margin
        self.temp = temp


    def train(self, epoch, sub_id, data_loader, optimizer, train_iters,
                        print_freq=1, vlad=True, loss_type='triplet'):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):
            inputs = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)
            with autocast():
                loss = self._forward(inputs, vlad, loss_type)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}-{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, sub_id, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))


    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        # imgs_size: batch_size*triplet_size*C*H*W
        return imgs.cuda(self.gpu)


    def _forward(self, inputs, vlad, loss_type):
        B, N, C, H, W = inputs.size()
        inputs = inputs.view(-1, C, H, W)

        outputs_pool, outputs_vlad = self.model(inputs)
        if (not vlad):
            # adopt VLAD layer for feature aggregation
            return self._get_loss(outputs_pool, loss_type, B, N)
        else:
            # adopt max pooling for feature aggregation
            return self._get_loss(outputs_vlad, loss_type, B, N)


    def _get_loss(self, outputs, loss_type, B, N):
        outputs = outputs.view(B, N, -1)
        L = outputs.size(-1)

        output_negatives = outputs[:, 2:]
        output_anchors = outputs[:, 0]
        output_positives = outputs[:, 1]

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist = - torch.cat((dist_pos, dist_neg), 1)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        elif (loss_type=='sare_ind'):
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss


class WAKDTrainer(object):
    #############################
    # Training module for
    # "Weak-supervised Visual Geo-localization via Attention-based Knowledge Distillation"
    #############################
    def __init__(self, model, model_cache, model_teacher, margin=0.3,
                    neg_num=10, gpu=None, temp=[0.07,]):
        super(WAKDTrainer, self).__init__()
        self.model = model
        self.model_cache = model_cache
        self.model_teacher = model_teacher
        self.margin = margin
        self.gpu = gpu
        self.neg_num = neg_num
        self.temp = temp


    def train(self, gen, epoch, sub_id, data_loader, optimizer, train_iters,
                    print_freq=1, lambda_soft=0.5, loss_type='sare_ind'):
        self.model.train()
        self.model_cache.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_hard = AverageMeter()
        losses_soft = AverageMeter()
        losses_kd = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):

            inputs_easy, inputs_diff = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            loss_hard, loss_soft = self._forward(inputs_easy, inputs_diff, loss_type, gen)
            loss_kd = self._get_kd_attention_loss(inputs_easy, inputs_diff) * 100

            loss = loss_hard + loss_soft*lambda_soft + loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_hard.update(loss_hard.item())
            losses_soft.update(loss_soft.item())
            losses_kd.update(loss_kd.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}-{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_hard {:.3f} ({:.3f})\t'
                      'Loss_soft {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})'
                      .format(epoch, sub_id, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_hard.val, losses_hard.avg,
                              losses_soft.val, losses_soft.avg,
                              losses_kd.val, losses_kd.avg))


    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        imgs_easy = imgs[:,:self.neg_num+2]
        imgs_diff = torch.cat((imgs[:,0].unsqueeze(1).contiguous(), imgs[:,self.neg_num+2:]), dim=1)
        return imgs_easy.cuda(self.gpu), imgs_diff.cuda(self.gpu)


    def _forward(self, inputs_easy, inputs_diff, loss_type, gen):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.view(-1, C, H, W)
        inputs_diff = inputs_diff.view(-1, C, H, W)

        sim_easy, vlad_anchors, vlad_pairs = self.model(inputs_easy)
        # vlad_anchors: B*1*9*L
        # vlad_pairs: B*(1+neg_num)*9*L
        with torch.no_grad():
            sim_diff_label, _, _ = self.model_cache(inputs_diff) # B*diff_pos_num*9*9
        sim_diff, _, _ = self.model(inputs_diff)

        if (gen==0):
            loss_hard = self._get_loss(vlad_anchors[:,0,0], vlad_pairs[:,0,0], vlad_pairs[:,1:,0], B, loss_type)
        else:
            loss_hard = 0
            for tri_idx in range(B):
                loss_hard += self._get_hard_loss(vlad_anchors[tri_idx,0,0].contiguous(), vlad_pairs[tri_idx,0,0].contiguous(), \
                                                vlad_pairs[tri_idx,1:], sim_easy[tri_idx,1:,0].contiguous().detach(), loss_type)
            loss_hard /= B

        log_sim_diff = F.log_softmax(sim_diff[:,:,0].contiguous().view(B,-1)/self.temp[0], dim=1)
        loss_soft = (- F.softmax(sim_diff_label[:,:,0].contiguous().view(B,-1)/self.temp[gen], dim=1).detach() * log_sim_diff).mean(0).sum()

        return loss_hard, loss_soft


    def _get_hard_loss(self, anchors, positives, negatives, score_neg, loss_type):
        # select the most difficult regions for negatives
        score_arg = score_neg.view(self.neg_num,-1).argmax(1)
        score_arg = score_arg.unsqueeze(-1).unsqueeze(-1).expand_as(negatives).contiguous()
        select_negatives = torch.gather(negatives,1,score_arg)
        select_negatives = select_negatives[:,0]

        return self._get_loss(anchors.unsqueeze(0).contiguous(), \
                            positives.unsqueeze(0).contiguous(), \
                            select_negatives.unsqueeze(0).contiguous(), 1, loss_type)


    def _get_loss(self, output_anchors, output_positives, output_negatives, B, loss_type):
        L = output_anchors.size(-1)

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)
            dist_neg = dist_neg.view(B, -1)

            # joint optimize
            dist = torch.cat((dist_pos, dist_neg), 1)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        elif (loss_type=='sare_ind'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)

            dist_neg = dist_neg.view(B, -1)

            # indivial optimize
            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss
    

    def _get_kd_loss(self, inputs_easy, inputs_diff):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.view(-1, C, H, W)
        inputs_diff = inputs_diff.view(-1, C, H, W)
        inputs_all = torch.cat([inputs_easy, inputs_diff], dim=0)

        _, x_student = self.model.module.base_model(inputs_all)
        vlad_x_student = self.model.module.net_vlad(x_student)
        # normalize
        vlad_x_student = F.normalize(vlad_x_student, p=2, dim=2)  # intra-normalization
        vlad_x_student = vlad_x_student.view(x_student.size(0), -1)  # flatten
        vlad_g_student = torch.mm(vlad_x_student, vlad_x_student.t())
        outputs_student = F.normalize(vlad_g_student, p=2, dim=1)  # L2 normalize
        with torch.no_grad():
            _, x_teacher = self.model_teacher.module.base_model(inputs_all)
            vlad_x_teacher = self.model_teacher.module.net_vlad(x_teacher)
            # normalize
            vlad_x_teacher = F.normalize(vlad_x_teacher, p=2, dim=2)  # intra-normalization
            vlad_x_teacher = vlad_x_teacher.view(x_teacher.size(0), -1)  # flatten
            vlad_g_teacher = torch.mm(vlad_x_teacher, vlad_x_teacher.t())
            outputs_teacher = F.normalize(vlad_g_teacher, p=2, dim=1)  # L2 normalize
        return F.mse_loss(outputs_student, outputs_teacher)


    def _get_kd_attention_loss(self, inputs_easy, inputs_diff):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.view(-1, C, H, W)
        inputs_diff = inputs_diff.view(-1, C, H, W)
        inputs_all = torch.cat([inputs_easy, inputs_diff], dim=0)

        _, x_student = self.model.base_model(inputs_all)

        # attention map similarity
        activations_student = torch.maximum(x_student, torch.tensor([0.0], device=x_student.device))
        attentions_inputs_student = activations_student.detach()
        attentions_inputs_student.requires_grad = True
        pool_inputs_student = torch.mean(attentions_inputs_student, dim=(2,3))
        pool_outputs_student = F.softmax(pool_inputs_student, dim=1)
        pre_class_student = torch.max(pool_outputs_student, dim=1, keepdim=True).values
        pre_class_student.backward(torch.tensor([[1.0] for _ in range(pre_class_student.shape[0])], device=pre_class_student.device))
        inputs_grads_student = torch.mean(attentions_inputs_student.grad, dim=(2,3), keepdim=True)
        attentions_student = activations_student * inputs_grads_student.detach()
        attentions_student = torch.mean(attentions_student, dim=1)
        attentions_student = torch.maximum(attentions_student, torch.tensor([0.0], device=attentions_student.device)) / \
                             torch.max(torch.max(attentions_student, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        attentions_student = attentions_student.view(attentions_student.size(0), -1)
        attentions_g_student = torch.mm(attentions_student, attentions_student.t())
        attentions_g_student = F.normalize(attentions_g_student, p=2, dim=1)

        # final feature similarity
        vlad_x_student = self.model.net_vlad(x_student)
        # normalize
        vlad_x_student = F.normalize(vlad_x_student, p=2, dim=2)  # intra-normalization
        vlad_x_student = vlad_x_student.view(x_student.size(0), -1)  # flatten
        vlad_g_student = torch.mm(vlad_x_student, vlad_x_student.t())
        outputs_student = F.normalize(vlad_g_student, p=2, dim=1)  # L2 normalize
        with torch.no_grad():
            _, x_teacher = self.model_teacher.base_model(inputs_all)
            vlad_x_teacher = self.model_teacher.net_vlad(x_teacher)
            # normalize
            vlad_x_teacher = F.normalize(vlad_x_teacher, p=2, dim=2)  # intra-normalization
            vlad_x_teacher = vlad_x_teacher.view(x_teacher.size(0), -1)  # flatten
            vlad_g_teacher = torch.mm(vlad_x_teacher, vlad_x_teacher.t())
            outputs_teacher = F.normalize(vlad_g_teacher, p=2, dim=1)  # L2 normalize
            
            # attention map similarity
            activations_teacher = torch.maximum(x_teacher,  torch.tensor([0.0], device=x_teacher.device))
        attentions_inputs_teacher = activations_teacher.detach()
        attentions_inputs_teacher.requires_grad = True
        pool_inputs_teacher = torch.mean(attentions_inputs_teacher, dim=(2,3))
        pool_outputs_teacher = F.softmax(pool_inputs_teacher, dim=1)
        pre_class_teacher = torch.max(pool_outputs_teacher, dim=1, keepdim=True).values
        pre_class_teacher.backward(torch.tensor([[1.0] for _ in range(pre_class_teacher.shape[0])], device=pre_class_teacher.device))
        with torch.no_grad():
            inputs_grads_teacher = torch.mean(attentions_inputs_teacher.grad, dim=(2,3), keepdim=True)
            attentions_teacher = activations_teacher * inputs_grads_teacher.detach()
            attentions_teacher = torch.mean(attentions_teacher, dim=1)
            attentions_teacher = torch.maximum(attentions_teacher, torch.tensor([0.0], device=attentions_teacher.device)) / \
                                    torch.max(torch.max(attentions_teacher, dim=2, keepdim=True).values, dim=1, keepdim=True).values
            attentions_teacher = attentions_teacher.view(attentions_teacher.size(0), -1)
            attentions_g_teacher = torch.mm(attentions_teacher, attentions_teacher.t())
            attentions_g_teacher = F.normalize(attentions_g_teacher, p=2, dim=1)

        return F.mse_loss(outputs_student, outputs_teacher) + F.mse_loss(attentions_g_student, attentions_g_teacher)


    def _get_kd_activation_loss(self, inputs_easy, inputs_diff):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.view(-1, C, H, W)
        inputs_diff = inputs_diff.view(-1, C, H, W)
        inputs_all = torch.cat([inputs_easy, inputs_diff], dim=0)

        _, x_student = self.model.base_model(inputs_all)

        # activation map similarity
        activations_student = F.normalize(x_student, p=2, dim=2)
        activations_student = activations_student.view(activations_student.size(0), -1)
        activations_g_student = torch.mm(activations_student, activations_student.t())
        activations_g_student = F.normalize(activations_g_student, p=2, dim=1)

        # final feature similarity
        vlad_x_student = self.model.net_vlad(x_student)
        # normalize
        vlad_x_student = F.normalize(vlad_x_student, p=2, dim=2)  # intra-normalization
        vlad_x_student = vlad_x_student.view(x_student.size(0), -1)  # flatten
        vlad_g_student = torch.mm(vlad_x_student, vlad_x_student.t())
        outputs_student = F.normalize(vlad_g_student, p=2, dim=1)  # L2 normalize
        with torch.no_grad():
            _, x_teacher = self.model_teacher.base_model(inputs_all)

            # activation map similarity
            activations_teacher= F.normalize(x_teacher, p=2, dim=2)
            activations_teacher = activations_teacher.view(activations_teacher.size(0), -1)
            activations_g_teacher = torch.mm(activations_teacher, activations_teacher.t())
            activations_g_teacher = F.normalize(activations_g_teacher, p=2, dim=1)

            # final feature similarity
            vlad_x_teacher = self.model_teacher.net_vlad(x_teacher)
            # normalize
            vlad_x_teacher = F.normalize(vlad_x_teacher, p=2, dim=2)  # intra-normalization
            vlad_x_teacher = vlad_x_teacher.view(x_teacher.size(0), -1)  # flatten
            vlad_g_teacher = torch.mm(vlad_x_teacher, vlad_x_teacher.t())
            outputs_teacher = F.normalize(vlad_g_teacher, p=2, dim=1)  # L2 normalize

        return F.mse_loss(outputs_student, outputs_teacher) + F.mse_loss(activations_g_student, activations_g_teacher)
