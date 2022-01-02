clc;
clear all;
close all;

outfolder = 'HOIP_TopoFeatures_S3';
    if ~isdir(outfolder)
        mkdir(outfolder)
    end

MLfeatures={'B','X','C','H','N','O',...
			'BX','BC','BH','BN','BO','XC','XH','XN','XO',...
			'BXC','BXH','BXN','BXO',...
			'BXCH','BXCN','BXCO','BXHN','BXHO','BXNO',...
			'BXCHN','BXCHO','BXCNO','BXHNO','BXCHNO'};
			
nfeatures =30;
ndata = 1346;
for fno = 1:nfeatures
	
	if fno==1
		filtration_size=15.0;
		fpath = strcat('./HOIP_gudhiOut_Bsite/','hoip_Bsite_',MLfeatures{fno});
    elseif fno==2
		filtration_size=15.0;
		fpath = strcat('./HOIP_gudhiOut_Xsite/','hoip_Xsite_',MLfeatures{fno});
    elseif (fno>2 && fno<7)
        filtration_size=15.0;
		fpath = strcat('./HOIP_gudhiOut_Asite/','hoip_Asite_',MLfeatures{fno});
	elseif (fno>6 && fno<20)
		filtration_size=15.0;
		fpath = strcat('./HOIP_gudhiOut_',MLfeatures{fno},'/','hoip_',MLfeatures{fno},'_');
	else
		filtration_size=10.0;
		fpath = strcat('./HOIP_gudhiOut_',MLfeatures{fno},'/','hoip_',MLfeatures{fno},'_');
	end
	
	nres=100;
	dcel=filtration_size/nres;
	
	pbf_b0 = zeros(nres,ndata);
	for fnd=1:ndata
		indata=strcat(fpath,num2str(fnd),'_b0','.txt');
		PHdata= load(indata);

		beta0 = PHdata;
		beout=beta0;
		m1=size(beout,1);

		x=zeros(nres,1);
		for irs=1:1:nres    
			xi=(irs-1)*dcel;
			x(irs)=xi;
		end 	
		for irs=1:1:nres     
			for i1=1:m1
				a1=beout(i1,1);
				b1=beout(i1,2); 
				if(x(irs)>=a1 && x(irs) <=b1)
					pbf_b0(irs,fnd)=pbf_b0(irs,fnd)+1;
				end
			end
        end
	end
	
	pbf_b1 = zeros(nres,ndata);		
	for fnd=1:ndata	
		indatb = strcat(fpath,num2str(fnd),'_b1','.txt');
		PHdata = load(indatb);

		beta1 = PHdata;
		beout=beta1;
		m1=size(beout,1);

		x=zeros(nres,1);
		for irs=1:1:nres    
			xi=(irs-1)*dcel;
			x(irs)=xi;
		end
		% 	
		for irs=1:1:nres     
			for i1=1:m1
				a1=beout(i1,1);
				b1=beout(i1,2); 
				if(x(irs)>=a1 && x(irs) <=b1)
					pbf_b1(irs,fnd)=pbf_b1(irs,fnd)+1;
				end
			end
		end
	end
	
	pbf_b2 = zeros(nres,ndata);		
	for fnd=1:ndata	
		indatc = strcat(fpath,num2str(fnd),'_b2','.txt');
		PHdata = load(indatc);

		beta2 = PHdata;
		beout=beta2;
		m1=size(beout,1);

		x=zeros(nres,1);
		for irs=1:1:nres    
			xi=(irs-1)*dcel;
			x(irs)=xi;
		end
		% 	
		for irs=1:1:nres     
			for i1=1:m1
				a1=beout(i1,1);
				b1=beout(i1,2); 
				if(x(irs)>=a1 && x(irs) <=b1)
					pbf_b2(irs,fnd)=pbf_b2(irs,fnd)+1;
				end
			end
		end
	end
	
	fname=strcat('./',outfolder,'/','hoip_',MLfeatures{fno},'_betti0','.txt');
	dlmwrite(fname,pbf_b0)
	fname=strcat('./',outfolder,'/','hoip_',MLfeatures{fno},'_betti1','.txt');
	dlmwrite(fname,pbf_b1)
	fname=strcat('./',outfolder,'/','hoip_',MLfeatures{fno},'_betti2','.txt');
	dlmwrite(fname,pbf_b2)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for fno = 1:nfeatures
	finb0 = strcat('./HOIP_TopoFeatures_S3/','hoip_',MLfeatures{fno},'_betti0.txt');
	b0= load(finb0);
	[rsize,csize]= size(b0);
	nb0 =zeros(rsize,csize);
	for ii = 1:rsize
		for jj = 1:csize
			val = b0(ii,jj);
			if (val == 1 || val == 0)
				nb0(ii,jj) = 0;
			else
				nb0(ii,jj) = 1;
			end
		end
	end
	b0req = sum(nb0,2);
	for kk = 1:rsize
		if(b0req(kk) ~= 0)
			b0mod(kk,:) = b0(kk,:);
		end
    end
    fnameb0 = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b0.txt');
	dlmwrite(fnameb0,b0mod)
    clear b0mod
end

for fno = 1:nfeatures
	finb1 = strcat('./HOIP_TopoFeatures_S3/','hoip_',MLfeatures{fno},'_betti1.txt');
	b1= load(finb1);
	[fsize,~]= size(b1);
	m=b1;
	newmat = m(all(b1==0,2),:);
	k=b1;
	k(all(b1==0,2),:)=[];
	b1mod = k;
    fnameb1 = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b1.txt');
	dlmwrite(fnameb1,b1mod)
    clear b1mod
end


for fno = 1:nfeatures
	finb2 = strcat('./HOIP_TopoFeatures_S3/','hoip_',MLfeatures{fno},'_betti2.txt');
	b2= load(finb2);
	[fsize,~]= size(b2);
	m=b2;
	newmat = m(all(b2==0,2),:);
	k=b2;
	k(all(b2==0,2),:)=[];
	b2mod = k;
    fnameb2 = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b2.txt');
	dlmwrite(fnameb2,b2mod)
    clear b2mod
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


startp=1;
for fno = 1:nfeatures
        fin = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b0','.txt');
        fein= load(fin);
        [endp,~] = size(fein);
		fedat = fein';    
		endp = startp+endp-1;
		Fdatab0(:,startp:endp)=fedat;
		startp=1+endp;  
end
dlmwrite('HOIP_PHb0_F30_S3.txt',Fdatab0)

startp=1;
for fno = 1:nfeatures
        fin = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b1','.txt');
        fein= load(fin);
        [endp,~] = size(fein);
		fedat = fein';    
		endp = startp+endp-1;
		Fdatab1(:,startp:endp)=fedat;
		startp=1+endp;  
end
dlmwrite('HOIP_PHb1_F30_S3.txt',Fdatab1)

startp=1;
for fno = 1:nfeatures
        fin = strcat('./HOIP_TopoFeatures_S3/hoip_',MLfeatures{fno},'_trun_b2','.txt');
        fein= load(fin);
        [endp,~] = size(fein);
		fedat = fein';    
		endp = startp+endp-1;
		Fdatab2(:,startp:endp)=fedat;
		startp=1+endp;  
end
dlmwrite('HOIP_PHb2_F30_S3.txt',Fdatab2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Hoip_B0 = load('HOIP_PHb0_F30_S3.txt');
Hoip_B1 = load('HOIP_PHb1_F30_S3.txt');
Hoip_B2 = load('HOIP_PHb2_F30_S3.txt');

Hoip_B0B1B2 = [Hoip_B0 Hoip_B1 Hoip_B2];
dlmwrite('HOIP_PHb0b1b2_F30_S3.txt',Hoip_B0B1B2)












