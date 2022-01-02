clc; 
clear all;
close all;

filename1=	'./PH_OUT_MAPbBr3_Cubic_CNPbX/Gudhi_Cubic_CNXPb_L5';
filename2=	'./PH_OUT_MAPbBr3_Ortho_CNPbX/Gudhi_Ortho_CNXPb_L5';
filename3=	'./PH_OUT_MAPbBr3_Tetra_CNPbX/Gudhi_Tetra_CNXPb_L5';

filename4=	'./PH_OUT_MAPbCl3_Cubic_CNPbX/Gudhi_Cubic_CNXPb_L5';
filename5=	'./PH_OUT_MAPbCl3_Ortho_CNPbX/Gudhi_Ortho_CNXPb_L5';
filename6=	'./PH_OUT_MAPbCl3_Tetra_CNPbX/Gudhi_Tetra_CNXPb_L5';

filename7=	'./PH_OUT_MAPbI3_Cubic_CNPbX/Gudhi_Cubic_CNXPb_L5';
filename8=	'./PH_OUT_MAPbI3_Ortho_CNPbX/Gudhi_Ortho_CNXPb_L5';
filename9=	'./PH_OUT_MAPbI3_Tetra_CNPbX/Gudhi_Tetra_CNXPb_L5';


nf=1000; % number of frames
filtration_size=10.0; % filteration size

nres=1000+1;
dcel=filtration_size/(nres-1);
x=zeros(nres,1);

   for irs=1:1:nres    
     xi=(irs-1)*dcel;
     x(irs)=xi;
   end

pbf_cube_Br=zeros(nres,nf,3);
pbf_orth_Br=zeros(nres,nf,3);
pbf_tetra_Br=zeros(nres,nf,3);

pbf_cube_Cl=zeros(nres,nf,3);
pbf_orth_Cl=zeros(nres,nf,3);
pbf_tetra_Cl=zeros(nres,nf,3);

pbf_cube_I=zeros(nres,nf,3);
pbf_orth_I=zeros(nres,nf,3);
pbf_tetra_I=zeros(nres,nf,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename1,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename1,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename1,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename1,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename1,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename1,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename1,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename1,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename1,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename1,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename1,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename1,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_cube_Br(irs,ifm,1)=pbf_cube_Br(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_Br(irs,ifm,2)=pbf_cube_Br(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_Br(irs,ifm,3)=pbf_cube_Br(irs,ifm,3)+1;
         end
     end   
   end   
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename2,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename2,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename2,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename2,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename2,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename2,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename2,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename2,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename2,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename2,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename2,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename2,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_orth_Br(irs,ifm,1)=pbf_orth_Br(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_Br(irs,ifm,2)=pbf_orth_Br(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_Br(irs,ifm,3)=pbf_orth_Br(irs,ifm,3)+1;
         end
     end   
   end   
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename3,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename3,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename3,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename3,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename3,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename3,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename3,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename3,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename3,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename3,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename3,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename3,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_tetra_Br(irs,ifm,1)=pbf_tetra_Br(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_Br(irs,ifm,2)=pbf_tetra_Br(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_Br(irs,ifm,3)=pbf_tetra_Br(irs,ifm,3)+1;
         end
     end   
   end   
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename4,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename4,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename4,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename4,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename4,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename4,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename4,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename4,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename4,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename4,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename4,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename4,itt); 
end

beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_cube_Cl(irs,ifm,1)=pbf_cube_Cl(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_Cl(irs,ifm,2)=pbf_cube_Cl(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_Cl(irs,ifm,3)=pbf_cube_Cl(irs,ifm,3)+1;
         end
     end   
   end   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename5,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename5,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename5,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename5,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename5,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename5,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename5,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename5,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename5,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename5,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename5,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename5,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_orth_Cl(irs,ifm,1)=pbf_orth_Cl(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_Cl(irs,ifm,2)=pbf_orth_Cl(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_Cl(irs,ifm,3)=pbf_orth_Cl(irs,ifm,3)+1;
         end
     end   
   end   
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename6,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename6,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename6,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename6,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename6,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename6,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename6,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename6,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename6,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename6,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename6,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename6,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_tetra_Cl(irs,ifm,1)=pbf_tetra_Cl(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_Cl(irs,ifm,2)=pbf_tetra_Cl(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_Cl(irs,ifm,3)=pbf_tetra_Cl(irs,ifm,3)+1;
         end
     end   
   end   

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename7,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename7,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename7,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename7,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename7,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename7,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename7,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename7,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename7,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename7,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename7,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename7,itt); 
end

beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_cube_I(irs,ifm,1)=pbf_cube_I(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_I(irs,ifm,2)=pbf_cube_I(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_cube_I(irs,ifm,3)=pbf_cube_I(irs,ifm,3)+1;
         end
     end   
   end   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename8,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename8,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename8,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename8,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename8,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename8,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename8,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename8,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename8,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename8,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename8,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename8,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_orth_I(irs,ifm,1)=pbf_orth_I(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_I(irs,ifm,2)=pbf_orth_I(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_orth_I(irs,ifm,3)=pbf_orth_I(irs,ifm,3)+1;
         end
     end   
   end   
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ifm=1:nf
   itt=ifm;
if(ifm<=9)
mybt0 = sprintf('%s_f%d_b0.txt',filename9,itt);
mybt1 = sprintf('%s_f%d_b1.txt',filename9,itt);
mybt2 = sprintf('%s_f%d_b2.txt',filename9,itt); 
elseif(ifm>9 && ifm <100)
mybt0 = sprintf('%s_f%2d_b0.txt',filename9,itt);
mybt1 = sprintf('%s_f%2d_b1.txt',filename9,itt);
mybt2 = sprintf('%s_f%2d_b2.txt',filename9,itt);     
elseif(ifm>99 && ifm <1000)
mybt0 = sprintf('%s_f%3d_b0.txt',filename9,itt);
mybt1 = sprintf('%s_f%3d_b1.txt',filename9,itt);
mybt2 = sprintf('%s_f%3d_b2.txt',filename9,itt); 
else
mybt0 = sprintf('%s_f%4d_b0.txt',filename9,itt);
mybt1 = sprintf('%s_f%4d_b1.txt',filename9,itt);
mybt2 = sprintf('%s_f%4d_b2.txt',filename9,itt); 
end
    
beta0=load(mybt0);
beta1=load(mybt1);
beta2=load(mybt2);

beout=beta0;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs) <=b1)
           pbf_tetra_I(irs,ifm,1)=pbf_tetra_I(irs,ifm,1)+1;
         end
     end   
   end
   
beout=beta1;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_I(irs,ifm,2)=pbf_tetra_I(irs,ifm,2)+1;
         end
     end   
   end   

beout=beta2;
m0=size(beout,1);
   for irs=1:1:nres     
     for i1=1:m0
         a1=beout(i1,1);
         b1=beout(i1,2); 
         if(x(irs)>=a1 && x(irs)<=b1)
           pbf_tetra_I(irs,ifm,3)=pbf_tetra_I(irs,ifm,3)+1;
         end
     end   
   end   

end

linw=2;
fs=30;
frs=500;
frd=1000;

nrestrn=300;

bettiBr=zeros(3*(frd-frs+1),nrestrn,3);
bettiCl=zeros(3*(frd-frs+1),nrestrn,3);
bettiI=zeros(3*(frd-frs+1),nrestrn,3);

for ndim=1:3
	if ndim==1
		spbn=101;
		epbn=400;
    elseif ndim==2
		spbn=201;
		epbn=500;
	else
		spbn=301;
		epbn=600;
	end
	
	
	bettiBr(1:frd-frs+1,1:nrestrn,ndim)=pbf_cube_Br(spbn:epbn,frs:frd,ndim)';
	bettiBr(frd-frs+2:2*(frd-frs+1),1:nrestrn,ndim)=pbf_orth_Br(spbn:epbn,frs:frd,ndim)';
	bettiBr((frd-frs+1)*2+1:3*(frd-frs+1),1:nrestrn,ndim)=pbf_tetra_Br(spbn:epbn,frs:frd,ndim)';
	
	
	bettiCl(1:frd-frs+1,1:nrestrn,ndim)=pbf_cube_Cl(spbn:epbn,frs:frd,ndim)';
	bettiCl(frd-frs+2:2*(frd-frs+1),1:nrestrn,ndim)=pbf_orth_Cl(spbn:epbn,frs:frd,ndim)';
	bettiCl((frd-frs+1)*2+1:3*(frd-frs+1),1:nrestrn,ndim)=pbf_tetra_Cl(spbn:epbn,frs:frd,ndim)';


	bettiI(1:frd-frs+1,1:nrestrn,ndim)=pbf_cube_I(spbn:epbn,frs:frd,ndim)';
	bettiI(frd-frs+2:2*(frd-frs+1),1:nrestrn,ndim)=pbf_orth_I(spbn:epbn,frs:frd,ndim)';
	bettiI((frd-frs+1)*2+1:3*(frd-frs+1),1:nrestrn,ndim)=pbf_tetra_I(spbn:epbn,frs:frd,ndim)';
end

 bettiBr2 = [bettiBr(:,:,1) bettiBr(:,:,2) bettiBr(:,:,3)];
 bettiCl2 = [bettiCl(:,:,1) bettiCl(:,:,2) bettiCl(:,:,3)];
 bettiI2 = [bettiI(:,:,1) bettiI(:,:,2) bettiI(:,:,3)];
 
betti=[bettiBr2; bettiCl2; bettiI2];


score=tsne(betti,'Algorithm','exact','Distance','euclidean');
% score=tsne(betti);
a=1;
for ii=1:4500
    cid(ii)=a;
    rm = mod(ii,500);
    if(rm==0)
        a=a+1;
    end
end
sid = ones(4500,1)*10;

figure;
plot(score(1:frd-frs+1,1),score(1:frd-frs+1,2),'k^'); 
hold on;
plot(score(frd-frs+2:2*(frd-frs+1),1),score(frd-frs+2:2*(frd-frs+1),2),'b^');
hold on; 
plot(score((frd-frs+1)*2+1:3*(frd-frs+1),1),score((frd-frs+1)*2+1:3*(frd-frs+1),2),'r^'); 
hold on;

plot(score((frd-frs+1)*3+1:4*(frd-frs+1),1),score((frd-frs+1)*3+1:4*(frd-frs+1),2),'mo'); 
hold on;
plot(score((frd-frs+1)*4+1:5*(frd-frs+1),1),score((frd-frs+1)*4+1:5*(frd-frs+1),2),'go'); 
hold on; 
plot(score((frd-frs+1)*5+1:6*(frd-frs+1),1),score((frd-frs+1)*5+1:6*(frd-frs+1),2),'yo'); 
hold on;

plot(score((frd-frs+1)*6+1:7*(frd-frs+1),1),score((frd-frs+1)*6+1:7*(frd-frs+1),2),'square','color','[0.0 1.0 1.0]'); 
hold on;
plot(score((frd-frs+1)*7+1:8*(frd-frs+1),1),score((frd-frs+1)*7+1:8*(frd-frs+1),2),'square','color','[0.9 0.4 0.1]'); 
hold on; 
plot(score((frd-frs+1)*8+1:9*(frd-frs+1),1),score((frd-frs+1)*8+1:9*(frd-frs+1),2),'square','color','[0.5 0.0 0.5]'); 
hold on;
set(gcf,'Position',[10 10 1000 1000])

















% [coeff,score,latent]=pca(betti);

% figure;
% plot(score(1:frd-frs+1,1),score(1:frd-frs+1,2),'b^','MarkerFaceColor','b'); 
% hold on;
% plot(score(frd-frs+2:2*(frd-frs+1),1),score(frd-frs+2:2*(frd-frs+1),2),'k^','MarkerFaceColor','k');
% hold on; 
% plot(score((frd-frs+1)*2+1:3*(frd-frs+1),1),score((frd-frs+1)*2+1:3*(frd-frs+1),2),'r^','MarkerFaceColor','r'); 
% hold on;

% plot(score((frd-frs+1)*3+1:4*(frd-frs+1),1),score((frd-frs+1)*3+1:4*(frd-frs+1),2),'mo','MarkerFaceColor','m'); 
% hold on;
% plot(score((frd-frs+1)*4+1:5*(frd-frs+1),1),score((frd-frs+1)*4+1:5*(frd-frs+1),2),'go','MarkerFaceColor','g'); 
% hold on; 
% plot(score((frd-frs+1)*5+1:6*(frd-frs+1),1),score((frd-frs+1)*5+1:6*(frd-frs+1),2),'yo','MarkerFaceColor','y'); 
% hold on;

% plot(score((frd-frs+1)*6+1:7*(frd-frs+1),1),score((frd-frs+1)*6+1:7*(frd-frs+1),2),'square','color','[0.0 1.0 1.0]','MarkerFaceColor',[0.0 1.0 1.0]); 
% hold on;
% plot(score((frd-frs+1)*7+1:8*(frd-frs+1),1),score((frd-frs+1)*7+1:8*(frd-frs+1),2),'square','color','[0.9 0.4 0.1]','MarkerFaceColor',[0.9 0.4 0.1]); 
% hold on; 
% plot(score((frd-frs+1)*8+1:9*(frd-frs+1),1),score((frd-frs+1)*8+1:9*(frd-frs+1),2),'square','color','[0.5 0.0 0.5]','MarkerFaceColor',[0.5 0.0 0.5]); 
% hold on;







