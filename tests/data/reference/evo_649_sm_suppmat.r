#### Adams and Collyer Example Data #########
#  Phenotypic Trajectory Analysis	

# CODE ASSUMES factor 1 = lineage, factor 2=evol. levels
rm(list=ls())
data<-read.csv("../evo_649_sm_example1.csv",header=T)
taxa<-as.factor(data[,1])
level<-as.factor(data[,2])
Y<-as.matrix(data[,-(1:2)])	## Removes factor columns from rest of data
taxaBylevel<-as.factor(paste(taxa,level))  #for LS means at each level
Y<-prcomp(Y)$x	#TO USE IF > 2 variables: for plotting PC axes at the end

#Basic data parameters (n=#spec, p=#ldmk, k=#dim)
n<-length(levels(taxa)) #was n.taxa
p<-length(levels(level))  #was n.level
k<-ncol(Y) #dimensions of phenotypic data

lm.full<-lm(Y~taxa*level,model=T,x=T,y=T,qr=T) 
yhat.full<-predict(lm.full)
summary(manova(lm.full))

	#For resid randomization (used later)
lm.red<-lm(Y ~ taxa + level, model=T,x=T,y=T,qr=T)   #just the covariate
yhat.red<-predict(lm.red)
res.red<-resid(lm.red)

lsmeans.obs<-NULL
for (i in 1:k){
  mean.temp<-tapply(yhat.full[,i],taxaBylevel,mean)
  lsmeans.obs<-cbind(lsmeans.obs,mean.temp)
}

arrayspecs<-function(A){
n.m<-NULL # n.m stands for "new" means
for(i in 1:n){
  temp<-as.matrix(A[((1+(i-1)*p):(i*p)),1:k])
  n.m<-cbind(n.m,temp)}
trajectories<-array(n.m,dim=c(p,k,n))
}

####Functions calculating Trajectory attributes

# Pathlength distance
pathdist<-function(M) {as.matrix(dist(M))}
trajsize<-function(M){
traj.pathdist<-array(0,dim=c(n,1))   		#loop across trajectories
for (i in 1:n){
  temp<-pathdist(M[,,i])
  for (j in 1:(p-1)){
    traj.pathdist[i]<-traj.pathdist[i]+temp[j,j+1]
  }
}
traj.size.dist<-as.matrix(dist(traj.pathdist))		#TrajSizeDiff
}

#trajectory direction 
orient<-function(M) {(svd(var(M))$v[1:k,1])} 	#find orientation
trajorient<-function(M){
traj.orient<-array(NA,dim=c(n,k))   		#loop across trajectories
check.1<-array(NA,dim=c(n))
for (i in 1:n){
  temp<-orient(M[,,i])
  traj.orient[i,]<-temp
  check.1[i]<-M[1,,i]%*%traj.orient[i,]  #check startingpoint location
  check.1[i]<-check.1[i]/abs(check.1[i])
  if(check.1[i]==-1) traj.orient[i,]<--1*traj.orient[i,]
}
options(warn=-1)				#b/c acos of 1 (e.g, on diagonal) yields warning
traj.ang.diff<-(180/pi)*acos(traj.orient%*%t(traj.orient))
#diag(traj.ang.diff)<-0
}

#trajectory shape
### GPA: following J. Claude 2008: Morphometrics in R
trans<-function(A){scale(A,scale=F)} 	##TRANSLATION
csize<-function(A)				##CSIZE
  {p<-dim(A)[1]
  size<-sqrt(sum(apply(A,2,var))*(p-1))
  list("centroid_size"=size,"scaled"=A/size)}
mshape<-function(A){apply(A,c(1,2),mean)}	#meanshape	

pPsup<-function(M1,M2){				## OPA rotation 1-->2
  k<-ncol(M1)
  Z1<-trans(csize(M1)[[2]])
  Z2<-trans(csize(M2)[[2]])
  sv<-svd(t(Z2)%*%Z1)
  U<-sv$v; V<-sv$u; Delt<-sv$d
  sig<-sign(det(t(Z1)%*%Z2))
  Delt[k]<-sig*abs(Delt[k]); V[,k]<-sig*V[,k]
  Gam<-U%*%t(V)
  beta<-sum(Delt)
  list(Mp1=beta*Z1%*%Gam,Mp2=Z2,rotation=Gam,scale=beta,
       df=sqrt(1-beta^2))}

pgpa<-function(A)
  {p<-dim(A)[1]; k<-dim(A)[2]; n<-dim(A)[3]  
  temp2<-temp1<-array(NA,dim=c(p,k,n)); Siz<-numeric(n)#; Qm2<-numeric(n)
  for (i in 1:n)
    {Acs<-csize(A[,,i])
    Siz[i]<-Acs[[1]]
    temp1[,,i]<-trans(Acs[[2]])}
  Qm1<-dist(t(matrix(temp1,k*p,n)))
  Q<-sum(Qm1); iter<-0
  while (abs(Q)> 0.00001)
    {for(i in 1:n){
      M<-mshape(temp1[,,-i])
      temp2[,,i]<-pPsup(temp1[,,i],M)[[1]]}
    Qm2<-dist(t(matrix(temp2,k*p,n)))
    Q<-sum(Qm1)-sum(Qm2)
    Qm1<-Qm2
    iter=iter+1
    temp1<-temp2}
  list("rotated"=temp2,"it.number"=iter,"Q"=Q,"intereucl.dist"=Qm2,"mshape"=
	csize(mshape(temp2))[[2]],"cent.size"=Siz)
}

## loop for GPA and shape distances
trajshape<-function(M){
  x<-pgpa(M)
  traj.shape.dist<-as.matrix(x$intereucl.dist) 
}

## TrajectorySummaryStat
sumstat<-function(M){
M<-as.dist(M)
x<-if(length(M)>1)(x=var(M)) else 0
}

##################MAIN LOOP################
traj.specs.obs<-arrayspecs(lsmeans.obs)
trajsize.obs<-trajsize(traj.specs.obs)
trajdir.obs<-trajorient(traj.specs.obs)
diag(trajdir.obs)<-0 #b/c some NA/N values on diagonal)
trajshape.obs<-trajshape(traj.specs.obs)
sumstatsize.obs<-sumstat(trajsize.obs)
sumstatdir.obs<-sumstat(trajdir.obs)
sumstatshape.obs<-sumstat(trajshape.obs)


### PERMUTATION PROCEDURE
permute<-999
line<-nrow(Y)
PSize<-POrient<-PShape<-array(1,dim=c(n,n))
PSumSize<-PSumOrient<-PSumShape<-1
for(i in 1:permute){
	line.rand<-sample(line,replace=FALSE)
	res.temp<-cbind(line.rand,res.red)
	z<-(order(line.rand))
	res.temp2<-as.matrix(res.temp[z,])
	res.p<-res.temp2[,-1]  # Rows of residuals are now randomized
	y.rand<-yhat.red+res.p
	yhat.rand<-predict(lm(y.rand~taxa*level,model=T,x=T,y=T,qr=T))

	lsmeans.rand<-NULL
	for (j in 1:k){
	  mean.temp<-tapply(yhat.rand[,j],taxaBylevel,mean)
	  lsmeans.rand<-cbind(lsmeans.rand,mean.temp)
	}
	traj.specs.rand<-arrayspecs(lsmeans.rand)
	trajsize.rand<-trajsize(traj.specs.rand)
	trajdir.rand<-trajorient(traj.specs.rand)
	diag(trajdir.rand)<-0 #b/c some NA/N values on diagonal)
	trajshape.rand<-trajshape(traj.specs.rand)
	sumstatsize.rand<-sumstat(trajsize.rand)
	sumstatdir.rand<-sumstat(trajdir.rand)
	sumstatshape.rand<-sumstat(trajshape.rand)
	for(j in 1:n){
	  for(jj in 1:n){
		PSize[j,jj]<-if(trajsize.rand[j,jj]>=trajsize.obs[j,jj]) 
			(PSize[j,jj]+1) else PSize[j,jj]
		POrient[j,jj]<-if(trajdir.rand[j,jj]>=trajdir.obs[j,jj]) 
			(POrient[j,jj]+1) else POrient[j,jj]
		PShape[j,jj]<-if(trajshape.rand[j,jj]>=trajshape.obs[j,jj]) 
			(PShape[j,jj]+1) else PShape[j,jj]
	  }
	}
	PSumSize<-if(sumstatsize.rand>=sumstatsize.obs) (PSumSize+1) else PSumSize
	PSumSOrient<-if(sumstatdir.rand>=sumstatdir.obs) (PSumOrient+1) else PSumOrient
	PSumShape<-if(sumstatshape.rand>=sumstatshape.obs) (PSumShape+1) else PSumShape
}  #end permute

for(j in 1:n){
  for(jj in 1:n){
	PSize[j,jj]<-PSize[j,jj]/(permute+1)
	POrient[j,jj]<-POrient[j,jj]/(permute+1)
	PShape[j,jj]<-PShape[j,jj]/(permute+1)
  }
}
PSumSize<-PSumSize/(permute+1)
PSumOrient<-PSumOrient/(permute+1)
PSumShape<-PSumShape/(permute+1)

trajsize.obs
PSize
trajdir.obs
POrient
trajshape.obs
PShape

######NOTE: summary statistics only valid for 2+ trajectories!!
# sumstatsize.obs
# PSumSize
# sumstatdir.obs
# PSumOrient
# sumstatshape.obs
# PSumShape
# 
# #########Plot Data and LS means 
# plot(Y[,1:2],type="n",xlab="PC I", ylab="PC II",asp=1)
# points(Y[,1:2])
# for (i in 1:n){		 	
#   for (j in 1:(p-1)){		
#     points(traj.specs.obs[(j:(j+1)),1,i],traj.specs.obs[(j:(j+1)),2,i],type="l",pch=21,col="black")    
#   }
# }
# for (i in 1:n){		 	
#   for (j in 2:(p-1)){		
#     points(traj.specs.obs[j,1,i],traj.specs.obs[j,2,i],pch=21,bg="gray",col="black",cex=1.5)
#   }
# }
# for (i in 1:n){
#   points(traj.specs.obs[1,1,i],traj.specs.obs[1,2,i],pch=21,bg="white",col="black",cex=1.5)
# }
# for (i in 1:n){
#   points(traj.specs.obs[p,1,i],traj.specs.obs[p,2,i],pch=21,bg="black",col="black",cex=1.5)
# }

