clear
clc
tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%    Input parameters   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CNN=[150528,150528,1,1,1,1; %Preprocessing
     3,64,224,224,3,1;
     64,128,224,224,3,1;
     128,128,112,112,3,1;
     128,128,112,112,3,1;
     128,256,56,56,3,1;
     256,256,56,56,3,1;
     256,256,56,56,3,1;
     256,512,28,28,3,1;
     512,512,28,28,3,1;
     512,512,28,28,3,1;
     512,512,14,14,3,1;
     512,512,14,14,3,1;
     512,512,14,14,3,1;
     512,4096,7,7,7,1;
     4096,4096,1,1,1,1;
     4096,1000,1,1,1,1;]; %N,M,R,C,K,S

%%%%%%%%%%%%%%%%%%%%%%%%%%%    Hardware Information  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NUM_DSP=floor(6833/5);
BRAM=8.17+33.75;  %MB
DATA_TYPE=4; %B;
DDR_BW=77; % GB/s
freq=250; %MHz



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NUM_layer,y]=size(CNN);
Exe_total=1e30;
Tm_final=NUM_DSP;
Tn_final=NUM_DSP;
Tr_final=zeros(1,NUM_layer);
Tc_final=zeros(1,NUM_layer);
BRAM_layer=zeros(1,NUM_layer);

for Tn=1:NUM_DSP
    for Tm=1:floor(NUM_DSP/Tn)
        Exe_min=ones(1,NUM_layer)*1e20;
        flag=0;
        cnt=0;
        for i=1:NUM_layer
            N=CNN(i,1);
            M=CNN(i,2);
            R=CNN(i,3);
            C=CNN(i,4);
            K=CNN(i,5);
            S=CNN(i,6);
            for Tr=1:R
                for Tc=1:C
                    B_in=Tn*(S*Tr+K-S)*(S*Tc+K-S)*DATA_TYPE;
                    B_wght=Tm*Tn*K^2*DATA_TYPE;
                    B_out=Tm*Tr*Tc*DATA_TYPE;
                    a_in=ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc);
                    a_wght=a_in;
                    a_out=ceil(M/Tm)*ceil(R/Tr)*ceil(C/Tc);
                    Comp_cyc=ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc)*(Tr*Tc*K*K);    %Cycle number
                    Data_access=a_in*B_in+a_wght*B_wght+a_out*B_out;        %Data size：B
                    Comm_cyc=(Data_access/(1024^3)/DDR_BW)*(freq*1e6);
                    BRAM_req=2*(B_in+B_wght+B_out)/(1024^2);
                    if(BRAM_req<=BRAM)
                        Exe_cyc=max(Comm_cyc,Comp_cyc);
                        if(Exe_cyc<Exe_min(1,i))
                            Exe_min(1,i)=Exe_cyc;
                            Tr_final(1,i)=Tr;
                            Tc_final(1,i)=Tc;
                            BRAM_layer(1,i)=BRAM_req;
                        end
                    end
                end
            end

            if(Exe_min(1,i)~=1e20)
                cnt=cnt+1;
                if(cnt==NUM_layer)
                    flag=1;
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
