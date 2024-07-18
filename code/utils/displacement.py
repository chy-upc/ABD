import torch
from einops import rearrange
import random

def ABD_R(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    for i in range(args.labeled_bs):
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
        patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
        a_ori = patches_mean_1_top4_indices[i, a]
        c_ori = patches_mean_2_top4_indices[i, c]

        max_patch_1 = image_patch_2[i][c_ori]  
        image_patch_1[i][b] = max_patch_1  
        max_patch_2 = image_patch_1[i][a_ori]
        image_patch_2[i][d] = max_patch_2 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last

def ABD_R_BCP(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ',p1=args.patch_size, p2=args.patch_size)  # torch.Size([12, 224, 224])
    image_patch_2 = rearrange(net_input_2.squeeze(1),'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a_ori]
            image_patch_2[i][d] = max_patch_2
        else:
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a]
            image_patch_2[i][d] = max_patch_2
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def ABD_I(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # ABD-I Bidirectional Displacement Patch
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)
    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1)
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1)
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1)
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1)
    for i in range(args.labeled_bs): 
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2[i][h[i]]  
            image_patch_supervised_1[i][e[i]] = min_patch_supervised_1
            min_patch_supervised_2 = image_patch_supervised_1[i][f[i]]
            image_patch_supervised_2[i][g[i]] = min_patch_supervised_2

            min_label_supervised_1 = label_patch_supervised_2[i][h[i]]
            label_patch_supervised_1[i][e[i]] = min_label_supervised_1
            min_label_supervised_2 = label_patch_supervised_1[i][f[i]]
            label_patch_supervised_2[i][g[i]] = min_label_supervised_2
    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(label_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    return image_patch_supervised_last, label_patch_supervised_last

