from __future__ import print_function
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier
import classifier2
import sys
import model

import numpy as np
import myutils
from myutils import classifierSSL, Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=170, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=800, help='number of all classes')
parser.add_argument('--beta4', type=int, default=0.01, help='parameters-l2')


parser.add_argument('--beta2', type=int, default=0.01, help='DIV')
parser.add_argument('--beta3', type=int, default=0.01, help='SELF')
parser.add_argument('--NUM', type=int, default=500, help='number of features by div')




sys.stdout = Logger('log/'+time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())+'.log')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

dictionary = np.load('dictionary/'+opt.dataset+'-K-means.npy',allow_pickle=True).item()
clsSSL=classifierSSL(opt.resSize,len(dictionary.keys())+1)
optimizerSSL = optim.Adam(clsSSL.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
crossentropyloss=nn.CrossEntropyLoss()


# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()


input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    clsSSL.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num,dictionary,opt):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize).cuda()
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)

    syn_noise = torch.FloatTensor(num, opt.nz)
    num_c=opt.NUM
    syn_noise_c = torch.FloatTensor(num_c, opt.nz)
    syn_att_c = torch.FloatTensor(num_c, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_noise_c = syn_noise_c.cuda()
        syn_att_c=syn_att_c.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data)
        syn_label.narrow(0, i * num, num).fill_(iclass)

    syn_label=syn_label.numpy().flatten()
    syn_label=syn_label.tolist()

    if opt.beta2 != 0 or (opt.beta3 !=0 and opt.beta2 != 0):
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att_c.copy_(iclass_att.repeat(num_c, 1))
            c=np.ones(num_c)
            c=c*int(iclass)
            c=c.tolist()
            for j in dictionary.keys():
                syn_noise_c.normal_(0, 1)
                num += 1
                attchange, _ = myutils.changeAtt(syn_att_c, j, dictionary[j])
                with torch.no_grad():
                    output = netG(Variable(syn_noise_c, volatile=True), Variable(attchange, volatile=True))
                syn_feature=torch.cat((syn_feature,output.data),0)
                syn_label.extend(c)

    syn_label=np.asarray(syn_label)
    syn_label=torch.from_numpy(syn_label).long()

    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty






# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100,
                                     opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False
h=0
for epoch in range(opt.nepoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            # sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            #criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            # fake_norm = fake.norm()
            # sparse_fake = fake.item().eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            #criticD_fake.backward(one)

            input_ress =input_res
            fakes=fake
            input_attvs=input_attv
            criticD_fake_ct=0
            num=0
            gradient_penalty_ct= 0
            for i in dictionary.keys():
                num+=1
                attchange,_= myutils.changeAtt(input_attv, i, dictionary[i])
                fake_c = netG(noisev, attchange)
                criticD_fake_c = netD(fake_c.detach(), input_attv)
                criticD_fake_ct += criticD_fake_c.mean()
                input_ress=torch.cat((input_ress, input_res), 0)
                fakes=torch.cat((fakes, fake_c), 0)
                input_attvs=torch.cat((input_attvs,input_attv),0)

            criticD_fake_ct=criticD_fake_ct/num
            input_ress=input_ress[opt.batch_size:,:]
            fakes=fakes[opt.batch_size:,:]
            input_attvs=input_attvs[opt.batch_size:,:]


            gradient_penalty_c = calc_gradient_penalty(netD, input_ress, fakes.data, input_attvs)


            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            # gradient_penalty.backward()
            lossG_c=criticD_fake_ct-criticD_real+gradient_penalty_c
            lossG = criticD_fake - criticD_real + gradient_penalty + opt.beta2*lossG_c

            lossG.backward()
            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation

        netG.zero_grad()
        clsSSL.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
        errG = G_cost + opt.cls_weight * c_errG

        #input_ress = input_res
        fakes = fake
        #input_attvs = input_attv
        G_cost_ct = 0
        labels=input_label.clone().detach()
        num = 0
        labels_SSL_0=np.zeros(input_attv.size(0))
        labels_SSL_0=labels_SSL_0.flatten().tolist()
        labels_SSL=[]

        for i in dictionary.keys():
            num += 1
            attchange, label_SSL = myutils.changeAtt(input_attv, i, dictionary[i])
            fake_c = netG(noisev, attchange)
            criticD_fake_c = netD(fake_c.detach(), input_attv)
            G_cost_ct += criticD_fake_c.mean()
            #input_ress = torch.cat((input_ress, input_res), 0)
            fakes = torch.cat((fakes, fake_c), 0)
            #input_attvs = torch.cat((input_attvs, input_attv), 0)
            labels = torch.cat((labels,input_label),0)
            labels_SSL.extend(label_SSL)
        G_cost_ct =- G_cost_ct / num
        #input_ress = input_ress[opt.batch_size:, :]
        fakes = fakes[opt.batch_size:, :]
        #input_attvs = input_attvs[opt.batch_size:, :]
        labels = labels[opt.batch_size:].view(-1)

        c_errG_c = cls_criterion(pretrain_cls.model(fakes), Variable(labels))
        errG_c = G_cost_ct + opt.cls_weight * c_errG_c

        fakes=torch.cat((fakes,fake))
        labels_SSL.extend(labels_SSL_0)
        labels_SSL=np.asarray(labels_SSL).flatten()
        labels_SSL=torch.from_numpy(labels_SSL).cuda().long()
        pre = clsSSL(fakes)

        crossentropyloss_ssl= crossentropyloss(pre, labels_SSL)

        l2loss = 0
        for param in netG.parameters():
            l2loss += torch.norm(param)



        #errG.backward()
        errG_t=errG+opt.beta2*errG_c+ opt.beta3*crossentropyloss_ssl+opt.beta4*l2loss
        errG_t.backward()
        optimizerG.step()
        optimizerSSL.step()

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f,l2loss:%.4f'
          % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item(),l2loss.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num,dictionary,opt)
        train_X = torch.cat((data.train_feature.cuda(), syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 100, opt.syn_num,
                                     True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if h<cls.H:
            h=cls.H
            print("best H is {0}".format(h))
            torch.save(netG,"model/unseen"+str(cls.acc_unseen)+"seen"+str(cls.acc_seen)+"H"+str(cls.H)+str(opt.dataset)+".pkl")
    # Zero-shot learning

    # reset G to training mode
    netG.train()