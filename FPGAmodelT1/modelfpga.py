#-*- coding: UTF-8 -*-
import math
import numpy as np
import time

CNN=np.array([
    [16, 3, 620, 460,3,1],#Conv_in
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [16, 16, 310, 230,3,2],
    [16, 32, 310, 230,3,1],#Downsample
    [32, 32, 155, 115,3,2],
    [32, 64, 155, 115,3,1],#Downsample
    [32, 16, 310, 230,3,1],
    [48, 16, 310, 230,3,1],
    [64, 16, 310, 230,3,1],
    [80, 16, 310, 230,3,1],
    [96, 32, 310, 230,1,1],#RBD
    [16, 16, 310, 230,3,2],
    [16, 32, 310, 230,3,1],#Downsample
    [32, 16, 310, 230,3,1],
    [48, 16, 310, 230,3,1],
    [64, 16, 310, 230,3,1],
    [80, 16, 310, 230,3,1],
    [96, 32, 310, 230,1,1],#RBD
    [16, 16, 310, 230,3,2],
    [16, 32, 310, 230,3,1],#Downsample
    [64, 16, 155, 115,3,1],
    [80, 16, 155, 115,3,1],
    [96, 16, 155, 115,3,1],
    [112, 16, 155, 115,3,1],
    [128, 64, 155, 115,1,1],#RBD
    [32, 32, 155, 115,3,2],
    [32, 64, 155, 115,3,1],#Downsample
    [64, 16, 155, 115,3,1],
    [80, 16, 155, 115,3,1],
    [96, 16, 155, 115,3,1],
    [112, 16, 155, 115,3,1],
    [128, 64, 155, 115,1,1],#RBD
    [32, 32, 155, 115,3,2],
    [32, 64, 155, 115,3,1],#downsample
    [64, 16, 155, 115,3,1],
    [80, 16, 155, 115,3,1],
    [96, 16, 155, 115,3,1],
    [112, 16, 155, 115,3,1],
    [128, 64, 155, 115,1,1],#RBD
    [64, 16, 155, 115,3,1],
    [80, 16, 155, 115,3,1],
    [96, 16, 155, 115,3,1],
    [112, 16, 155, 115,3,1],
    [128, 64, 155, 115,1,1],#RBD
    [64, 16, 155, 115,3,1],
    [80, 16, 155, 115,3,1],
    [96, 16, 155, 115,3,1],
    [112, 16, 155, 115,3,1],
    [128, 64, 155, 115,1,1],#RBD
    [32, 16, 310, 230,3,1],
    [48, 16, 310, 230,3,1],
    [64, 16, 310, 230,3,1],
    [80, 16, 310, 230,3,1],
    [96, 32, 310, 230,1,1],#RBD
    [64, 64, 310, 230,3,2],
    [64, 32, 310, 230,3,1],#Upsample
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [32, 32, 620, 460,3,2],
    [32, 16, 620, 460,3,1],#Upsample
    [32, 16, 310, 230,3,1],
    [48, 16, 310, 230,3,1],
    [64, 16, 310, 230,3,1],
    [80, 16, 310, 230,3,1],
    [96, 32, 310, 230,1,1],#RBD
    [64, 64, 310, 230,3,2],
    [64, 32, 310, 230,3,1],#Upsample
    [32, 16, 310, 230,3,1],
    [48, 16, 310, 230,3,1],
    [64, 16, 310, 230,3,1],
    [80, 16, 310, 230,3,1],
    [96, 32, 310, 230,1,1],#RBD
    [64, 64, 310, 230,3,2],
    [64, 32, 310, 230,3,1],#Upsample
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [32, 32, 620, 460,3,2],
    [32, 16, 620, 460,3,1],#Upsample
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [32, 32, 620, 460,3,2],
    [32, 16, 620, 460,3,1],#Upsample
    [16, 16, 620, 460,3,1],
    [32, 16, 620, 460,3,1],
    [48, 16, 620, 460,3,1],
    [64, 16, 620, 460,3,1],
    [80, 16, 620, 460,1,1],#RBD
    [16, 3, 620, 460,3,1]#Conv_out

],dtype=np.int64) #N,M,C,R,K,S

N_max=np.max(CNN[:,0])
M_max=np.max(CNN[:,1])

NUM_layer=CNN.shape[0]
NUM_DSP=math.floor(6833/5)
BRAM=8.17+33.75  #MB
DATA_TYPE=4 #B;
DDR_BW=77 # GB/s
freq=250 #MHz

Exe_total=0
Tm_final=1
Tn_final=1
start_all=0
start_time = time.time()
for Tn in range (1,min(N_max+1,NUM_DSP)):
    for Tm in range (1,min(M_max+1,math.floor(NUM_DSP/Tn))):
        min_per_layer=np.zeros((NUM_layer),dtype=np.float64)
        layer_index=0
        for i in range (0,NUM_layer):
            start = 0
            N=CNN[i][0]
            M=CNN[i][1]
            C=CNN[i][2]
            R=CNN[i][3]
            K=CNN[i][4]
            S=CNN[i][5]
            for Tr in range (1,R+1):
                Tc=C
                #for Tc in range (1,C+1):
                B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE
                B_wght=Tm*Tn*K*K*DATA_TYPE
                B_out=Tm*Tr*Tc*DATA_TYPE
                BRAM_req = 2 * float(B_in + B_wght + B_out) / (1e6)
                if (BRAM_req > BRAM):
                    break
                a_in=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)
                a_wght=a_in
                a_out=math.ceil(M/Tm)*math.ceil(R/Tr)*math.ceil(C/Tc)
                Comp_cyc=np.float64(math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)*(Tr*Tc*K*K))#Cycle number
                Data_access=np.float64(a_in*B_in+a_wght*B_wght+a_out*B_out)#Data size：B
                Comm_cyc=(Data_access/(1e9*DDR_BW))*(freq*1e6)
                Exe_cyc=max(Comm_cyc,Comp_cyc)
                if start==0:
                    min_per_layer[i]=Exe_cyc
                else:
                   min_per_layer[i]=min(min_per_layer[i],Exe_cyc)
                start=1
            if start==0:
                break
            layer_index+=1
        if layer_index<NUM_layer:
            break
        Exe_total=np.sum(min_per_layer)
        if start_all==0:
            Exe_min_sum=Exe_total
            Tm_final = Tm
            Tn_final = Tn
        else:
            if Exe_total<Exe_min_sum:
                Exe_min_sum = Exe_total
                Tm_final = Tm
                Tn_final = Tn
        start_all=1
print("Tm_final=",Tm_final)
print("Tn_final=",Tn_final)
print("Exe_total=",Exe_min_sum)
end_time = time.time() - start_time
print('search time is {0:.4f}'.format(end_time))

Tm=Tm_final
Tn=Tn_final
min_per_layer=np.zeros((NUM_layer))
for i in range (0,NUM_layer):
    start = 0
    N=CNN[i][0]
    M=CNN[i][1]
    C=CNN[i][2]
    R=CNN[i][3]
    K=CNN[i][4]
    S=CNN[i][5]
    Tr_best=0
    Tc_best=0
    for Tr in range (1,R+1):
        Tc=C
        #for Tc in range (1,C+1):
        B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE
        B_wght=Tm*Tn*K*K*DATA_TYPE
        B_out=Tm*Tr*Tc*DATA_TYPE
        BRAM_req = 2 * float(B_in + B_wght + B_out) / (1e6)
        if (BRAM_req > BRAM):
            break
        a_in=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)
        a_wght=a_in
        a_out=math.ceil(M/Tm)*math.ceil(R/Tr)*math.ceil(C/Tc)
        Comp_cyc=np.float64(math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)*(Tr*Tc*K*K))#Cycle number
        Data_access=np.float64(a_in*B_in+a_wght*B_wght+a_out*B_out)#Data size：B
        Comm_cyc=(Data_access/(1e9*DDR_BW))*(freq*1e6)
        # if start==0:
        #     print(B_in,B_wght,B_out,a_in,a_wght,a_out,Data_access)
        # print(Comm_cyc,Comp_cyc)
        Exe_cyc=max(Comm_cyc,Comp_cyc)
        if start==0:
            min_per_layer[i]=Exe_cyc
            Tr_best=Tr
            Tc_best=Tc
        else:
            if Exe_cyc<min_per_layer[i]:
                min_per_layer[i]=Exe_cyc
                Tr_best = Tr
                Tc_best = Tc
        start=1
    if start==0:
        break
    print("layer",i,"Tr,Tc",Tr_best,Tc_best)
print(np.sum(min_per_layer))


