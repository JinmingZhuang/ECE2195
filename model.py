#-*- coding: UTF-8 -*-
import math
import numpy as np

CNN=np.array([
     [1,6,32,32,5,1],
     [6,16,14,14,5,1],
     [16,120,5,5,5,1],
     [120,84,1,1,1,1],
    ]) #N,M,R,C,K,S

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
for Tn in range (1,min(N_max+1,NUM_DSP)):
    for Tm in range (1,min(M_max+1,math.floor(NUM_DSP/Tn))):
        min_per_layer=np.zeros((NUM_layer))
        layer_index=0
        for i in range (0,NUM_layer):
            start = 0
            N=CNN[i][0]
            M=CNN[i][1]
            R=CNN[i][2]
            C=CNN[i][3]
            K=CNN[i][4]
            S=CNN[i][5]
            for Tr in range (1,R+1):
                for Tc in range (1,C+1):
                    B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE
                    B_wght=Tm*Tn*K*K*DATA_TYPE
                    B_out=Tm*Tr*Tc*DATA_TYPE
                    BRAM_req = 2 * float(B_in + B_wght + B_out) / (1e6)
                    if (BRAM_req > BRAM):
                        break
                    a_in=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)
                    a_wght=a_in
                    a_out=math.ceil(M/Tm)*math.ceil(R/Tr)*math.ceil(C/Tc)
                    Comp_cyc=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)*(Tr*Tc*K*K)#Cycle number
                    Data_access=a_in*B_in+a_wght*B_wght+a_out*B_out#Data size：B
                    Comm_cyc=(float(Data_access)/(1e9*DDR_BW))*(freq*1e6)
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

Tm=Tm_final
Tn=Tn_final
min_per_layer=np.zeros((NUM_layer))
for i in range (0,NUM_layer):
    start = 0
    N=CNN[i][0]
    M=CNN[i][1]
    R=CNN[i][2]
    C=CNN[i][3]
    K=CNN[i][4]
    S=CNN[i][5]
    Tr_best=0
    Tc_best=0
    for Tr in range (1,R+1):
        for Tc in range (1,C+1):
            B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE
            B_wght=Tm*Tn*K*K*DATA_TYPE
            B_out=Tm*Tr*Tc*DATA_TYPE
            BRAM_req = 2 * float(B_in + B_wght + B_out) / (1e6)
            if (BRAM_req > BRAM):
                break
            a_in=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)
            a_wght=a_in
            a_out=math.ceil(M/Tm)*math.ceil(R/Tr)*math.ceil(C/Tc)
            Comp_cyc=math.ceil(M/Tm)*math.ceil(N/Tn)*math.ceil(R/Tr)*math.ceil(C/Tc)*(Tr*Tc*K*K)#Cycle number
            Data_access=a_in*B_in+a_wght*B_wght+a_out*B_out#Data size：B
            Comm_cyc=(float(Data_access)/(1e9*DDR_BW))*(freq*1e6)
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


