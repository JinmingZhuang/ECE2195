clear
clc
tic
NUM_layer=3;
CNN=[128,512,55,55,11,4;
     512,128,27,27,5,1;
     256,384,13,13,3,1;]; %N,M,R,C,K,S

NUM_DSP=floor(6833/5);
BRAM=20;  %MB
DATA_TYPE=4; %B;
DDR_BW=77; % GB/s
freq=250; %MHz

Exe_min=ones(1,NUM_layer)*1e20;
Exe_total=1e30;
for Tn=1:NUM_DSP
    for Tm=1:floor(NUM_DSP/Tn)
        flag=0;
        cnt=0;
        for i=1:NUM_layer
            N=CNN(i,1);
            M=CNN(i,2);
            R=CNN(i,3);
            C=CNN(i,4);
            K=CNN(i,5);
            S=CNN(i,6);
            if(Tm<M&&Tn<N)
                for Tr=1:R
                    for Tc=1:C
                        B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE;
                        B_wght=Tm*Tn*K^2*DATA_TYPE;
                        B_out=Tm*Tr*Tc*DATA_TYPE;
                        a_in=ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc);
                        a_wght=a_in;
                        a_out=ceil(M/Tm)*ceil(R/Tr)*ceil(C/Tc);
                        OP=2 * R * C * M * N * K * K;
                        Comp_cyc=ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc)*(Tr*Tc*K*K);    %Cycle number
                        Data_access=a_in*B_in+a_wght*B_wght+a_out*B_out;        %Data size：B
                        Comm_cyc=(Data_access/(1024^3)/DDR_BW)*(freq*1e6);
                        BRAM_req=2*(B_in+B_wght+B_out)/(1024^2);
                        if(BRAM_req<=BRAM)
                            Exe_cyc=max(Comm_cyc,Comp_cyc);
                            if(Exe_cyc<Exe_min(1,i))
                                Exe_min(1,i)=Exe_cyc;
                                cnt=cnt+1;
                                if(cnt==NUM_layer)
                                    flag=1;
                                end
                            end
                        end
                    end
                end
            end
        end
        if((sum(Exe_min)<=Exe_total)&&(flag==1))
            Tm_final=Tm;
            Tn_final=Tn;
            Exe_total=sum(Exe_min);
        end
    end
end
fprintf("Tm_final=%d \n",Tm_final);
fprintf("Tn_final=%d \n",Tn_final);
fprintf("Exe_total=%d \n",Exe_total);
toc