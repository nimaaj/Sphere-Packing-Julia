
using LsqFit
using JLD
using Shell
using Plots
using Polynomials
using Printf
using Combinatorics
using GenericLinearAlgebra
using PolynomialRoots
#using PlotlyJS
using Memoize
using NLsolve
using PrettyPrinting
using SpecialPolynomials
#region################################################# Define Blocks #################################################
setprecision(500)
function f(Δ,myc,nmax)
    x0=4*pi*(Δ-(myc-1)/12)
    LL=zeros(BigFloat,2*nmax);
    LL[1]=1;
    LL[1+1]=-x0+1;
    [LL[n+1]=1/n *((2*n - 1 - x0)* LL[n] - (n - 1) *LL[n - 1]) for n in 2:2*nmax-1];
    [LL[n+1] for n in 1:2:2*nmax-1];
end

function df(Δ,myc,nmax)
    x0=4*pi*(Δ-(myc-1)/12)
    dLL=zeros(BigFloat,2*nmax);
    dLL[1]=1;
    dLL[1+1]=-x0+2;
    [dLL[n+1]=1/n *((2*n - x0)* dLL[n] - (n) *dLL[n - 1]) for n in 2:2*nmax-2];
    [-4*pi*dLL[n+1] for n in 0:2:2*nmax-2];
end

function f0(myc,nmax)
    x0=4*pi*(-(myc-1)/12)
    x1=4*pi*(1-(myc-1)/12)
    x2=4*pi*(2-(myc-1)/12)
    LL1=zeros(BigFloat,2*nmax);
    LL2=zeros(BigFloat,2*nmax);
    LL3=zeros(BigFloat,2*nmax);
    LL1[1]=1;
    LL2[1]=1;
    LL3[1]=1;
    LL1[1+1]=-x0+1;
    LL2[1+1]=-x1+1;
    LL3[1+1]=-x2+1;
    for n in 2:2*nmax-1
        LL1[n+1]=1/n *((2*n - 1 - x0)* LL1[n] - (n - 1) *LL1[n - 1]);
        LL2[n+1]=1/n *((2*n - 1 - x1)* LL2[n] - (n - 1) *LL2[n - 1]);
        LL3[n+1]=1/n *((2*n - 1 - x2)* LL3[n] - (n - 1) *LL3[n - 1]);
    end
    t1=exp(2*pi*((myc-1)/12));
    t2=2*exp(-2*pi);
    t3=exp(-4*pi);
    [t1*(LL1[n+1]-t2*LL2[n+1]+t3*LL3[n+1]) for n in 1:2:2*nmax-1];
end
#endregion################################################# Define Blocks #################################################

#region################################################# Misc. Functions #################################################
function lowprec(x::Complex,prec)
    return Complex(BigFloat(x.re,precision=prec),BigFloat(x.im,precision=prec))
end
function lowprec(x,prec)
    return BigFloat(x,prec)
end
#endregion################################################# Misc. Functions #################################################

#region################################################# Define obj/jac #################################################
function obj!(F,x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    res=f0(myc,nmax)+ sum(Cs[i]*f(Rs[i],myc,nmax) for i=1:nmaxb2)
    [F[i]=res[i] for i=1:nmax]
end

function jac!(J,x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    res1=[Cs[i]*df(Rs[i],myc,nmax) for i=1:nmaxb2]
    res2=[f(Rs[i],myc,nmax) for i=1:nmaxb2]
    res3=vcat(res1,res2)
    [J[i,j]=res3[j][i] for i=1:nmax for j=1:nmax]
end


function testjac(x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    res1=[Cs[i]*df(Rs[i],myc,nmax) for i=1:nmaxb2]
    res2=[f(Rs[i],myc,nmax) for i=1:nmaxb2]
    res3=vcat(res1,res2);
    [res3[i][j] for i=1:nmax for j=1:nmax]
end
#testjac(iguess,curc)

function testobj(x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    println(Rs)
    println(Cs)
    f0(myc,nmax)+ sum(Cs[i]*f(Rs[i],myc,nmax) for i=1:nmaxb2)
end
#testobj(iguess,curc)
#endregion################################################# Define obj/jac #################################################


#region################################################# Define newton #################################################
function mynewton(ob,ja,guess,ϵ,maxiter,step)
    kk=1;
    p1=guess;
    ff1=ob(p1)
    dist=1;
    while kk<maxiter && dist>ϵ
        Δp=try -ja(p1)\ff1; catch mm; println("singular jac!"); return [false,false,false] end
        p1=p1+step * Δp
        println("--------------- Iteration ",kk," ---------------")
        ff1=ob(p1)
        dist=ff1'ff1
        println(lowprec(dist,5))
        kk+=1
    end
    if dist<ϵ println("Converged!")
    else
        println("failed")
    end
    return [p1,dist,dist<ϵ]
end

 function myobj(x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    f0(myc,nmax)+ sum(Cs[i]*f(Rs[i],myc,nmax) for i=1:nmaxb2)
end
function myjac(x,myc)
    nmax=length(x);
    nmaxb2=Int64(nmax/2);
    Rs=x[1:nmaxb2]
    Cs=x[nmaxb2+1:nmax]
    res1=[Cs[i]*df(Rs[i],myc,nmax) for i=1:nmaxb2]
    res2=[f(Rs[i],myc,nmax) for i=1:nmaxb2]
    res3=vcat(res1,res2)
    jacjac=zeros(BigFloat,nmax,nmax)
    [jacjac[i,j]=res3[j][i] for i=1:nmax for j=1:nmax];
    jacjac
end

# curc=BigFloat(4);
# mynmax=2;
# iguess=vcat([BigFloat(i+i/4) for i=1:mynmax],[BigFloat((10)^(i-1)) for i=1:mynmax])
# temp=mynewton(x->myobj(x,curc),x->myjac(x,curc),iguess,BigFloat(10^-10),20,1)

#endregion################################################# Define newton #################################################

#region################################################# guess functions #################################################
function rguess(unlist,newn)
    rlsts=[
    fit(
    [(newn-1)/(length(rootsave[curn])-1)*(i-1)+1 for i=(1:length(rootsave[curn]))],
    rootsave[curn],
    min(length(rootsave[curn])-1,22)
    ).(1:newn) 
    for curn in unlist]  #Rescale x-axis to fit domain of newn

    ftmodel(t, p) = p[1] .+ p[2]*t .+ p[3]*log.(t) .+ p[4]*(log.(t) .* log.(t) ) .+ p[5]*(log.(t) .* log.(t) .* log.(t) )
    p0 = [BigFloat(1) for i=1:5] 
    nlsts=[length(rootsave[curn])  for curn in unlist]
    [ftmodel(
            newn, 
            curve_fit(
                ftmodel,
                nlsts, 
                [rlsts[i][kk] for i=1:length(unlist)],
                p0).param
            ) for kk in 1:newn]
end

function cguess(unlist,newn)
    rlsts=[
    fit(
    [(newn-1)/(length(csave[curn])-1)*(i-1)+1 for i=(1:length(csave[curn]))],
    log.(csave[curn]),
    min(length(csave[curn])-1,22)
    ).(1:newn) 
    for curn in unlist]  #Rescale x-axis to fit domain of newn

    ftmodel(t, p) = p[1] .+ p[2]*t .+ p[3]*log.(t) .+ p[4]*(log.(t) .* log.(t) ) .+ p[5]*(log.(t) .* log.(t) .* log.(t) )
    p0 = [BigFloat(1) for i=1:5] 
    nlsts=[length(csave[curn])  for curn in unlist]
    [exp(ftmodel(
            newn, 
            curve_fit(
                ftmodel,
                nlsts, 
                [rlsts[i][kk] for i=1:length(unlist)],
                p0).param
            )) for kk in 1:newn]
end
#endregion################################################# guess functions #################################################

#region################################################# Initial solve #################################################
curc=BigFloat(4);
rootsave=[];
csave=[];
nlist=[];
for mynmax=2:10
    iguess=vcat([BigFloat(i+i/4) for i=1:mynmax],[BigFloat((10)^(i-1)) for i=1:mynmax])
    ntsln=mynewton(x->myobj(x,curc),x->myjac(x,curc),iguess,BigFloat(10^-50),30,1)
    push!(rootsave,ntsln[1][1:mynmax])
    push!(csave,ntsln[1][mynmax+1:2*mynmax])
    push!(nlist,mynmax)
end
#endregion################################################# Initial solve #################################################

#region################################################# Increase nmax to 60 #################################################
nlist
for mynmax=11:2:60
    iguess=vcat(rguess((max(1,length(nlist)-20):length(nlist)),mynmax),cguess((1:length(nlist)),mynmax))
    ntslnf=mynewton(x->myobj(x,curc),x->myjac(x,curc),iguess,BigFloat(10^-50),8500,1)
    #print(ntslnf)
    if !(ntslnf[3]) break end
    push!(rootsave,ntslnf[1][1:mynmax])
    push!(csave,ntslnf[1][mynmax+1:2*mynmax])
    push!(nlist,mynmax)
end

nlist
indi=34;
[exp(2*pi*(rootsave[indi][i]-(curc-1)/12))*csave[indi][i]     for i=1:10]
[exp(2*pi*(rootsave[indi][i]-(curc-1)/12))*csave[indi][i]-round(exp(2*pi*(rootsave[indi][i]-(curc-1)/12))*csave[indi][i])     for i=1:10]
#endregion################################################# Increase nmax to 60 #################################################


#region################################################# How well does the guess do? #################################################
nlist
sum(rootsave[34]-rguess((29:33),59))
sum(csave[34]-cguess((29:33),59))
#endregion################################################# How well does the guess do? #################################################

#region################################################# plots #################################################
plotlyjs(size=(1000,1000))
plot()

plot!([i for i=1:length(rootsave[11])],rootsave[11],seriestype = :scatter , legend=:topleft)
plot!([i for i=1:length(rootsave[12])],rootsave[12],seriestype = :scatter )
newn=16;
plot!(    (1:newn),    rguess((10:12),newn)    ,seriestype = :scatter, legend=:topleft)

plot()

plot!([i for i=1:length(log.(csave[11]))],log.(csave[11]) , legend=:topleft)
plot!([i for i=1:length(log.(csave[12]))],log.(csave[12]) )
newn=15;
plot!(    (1:newn),    log.(abs.(cguess((1:11),newn)))    , legend=:topleft)
#endregion################################################# plots #################################################

#region################################################# Increase nmax to 2000 #################################################

global mynmax=50
global noerror=0;
global maxchange=128;
global change=24;
while (mynmax<2000 && change>1)
    if noerror>6&& abs(2*change)<=maxchange
        change=2*change
        noerror=0
    end
    mynmax+=change;
    println("************** ", mynmax," ************** ", change," ************** ", noerror," ************** ")
    iguess=vcat(rguess((length(nlist)-5:length(nlist)),mynmax),cguess((1:length(nlist)),mynmax))
    ntsln=mynewton(x->myobj(x,curc),x->myjac(x,curc),iguess,BigFloat(10^-50),8500,1)
    solpassed=ntsln[3]

    if solpassed
        push!(rootsave,ntsln[1][1:mynmax])
        push!(csave,ntsln[1][mynmax+1:2*mynmax])
        push!(nlist,mynmax)
        noerror+=1;
    else
        noerror=0
        mynmax-=change
        change=Int64(round(change/2));
    end
end
#endregion################################################# Increase nmax to 300 #################################################


#region################################################# Save/Load Data #################################################
save("/home/modresults.jld2", "nlist",nlist, "rootsave",rootsave, "csave",csave)
loaddata=load("/home/modresults.jld2")
nlist=loaddata["nlist"];
rootsave=loaddata["rootsave"]
csave=loaddata["csave"]
csave[1]
#endregion################################################# Save/Load Data #################################################
