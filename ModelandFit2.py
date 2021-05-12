#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:02:40 2019

Simple toy model to assess g and ToM with respect to social and ecological complexity of a problem
Social complexity = increase in knowledge distribution and values and objectives
Ecological complexity = increase in non-linear dynamics as well as threshold and uncertainty

Other important system variables:
    Representational diversity = increases points of view, but also solution for the problem 
    Representational gaps = when two Groups represent the problem differently, 
    but sometimes they do not even know about it
@author: xxx, xxx

-----

in a nutshell group problem solving ability is given by social cmplx, ecol cmplx, rep. diversity, rep. gaps 
and g and ToM

Main Variables and parameters of the model
psol = probability of finding a solution
Rgap = representational gaps
Rdiv = representational diversity needed for maximum solution
tom  = Theory of Mind (ToM)
g    = general intelligence
scmp = social complexity
ecmp = ecological complexity

ecdivfac = how ecological complexity affects representational diversity needed for max solution note that here
            we use 10 as a scaling parameter, hence ecological complexity affects needs for rep.diversity based on a 
            logistic function that assumes values between 0 and 10.

Assumptions: 
Representational diversity necessary for maximal solution is given by the complexity of the problem. Both social and ecological complexity contribute to it, 
but ecological complexity is non-linearly realted (via ecdicvfac) to representational diversity needs.

Representational gaps are mediated by ToM and a factor (w_rgap) that mimicks added value to collective intelligence.
w_rgap grows with ToM non-linearly. That one need a certain amount of ToM for "collective intelligence factor" to kick in.

Main function: psol = f (Rgap, Rdiv, tom, g, scmp, ecmp)

equations rpresent the interplay between Rgap, Rdiv, tom and g and the probability of finding a solution.
scmp and ecmp represent the complexity of the problem at hand.

    

"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lmfit as lmf
import math


#parameter space
socomp   = list(np.arange(0, 11, 1))
ecomp    = list(np.arange(0, 11, 1))
ltom     = list (np.arange(0, 11, 1))
gen      = list (np.arange(0, 11, 1))

stg       = list (np.arange(1,11,1))
stt       = list (np.arange(1,11,1))

#Toy Model for IG - TOM interaction
def fgtomF (scmp, ecmp, stg, stt, g, tom):
    """
    MODEL PARAMETERS:
    ecmp = ecological complexity,
    scmp = social complexity, 
    stg  = coefficient for g effect on complexty and diversty
    stt  = coefficient for ToM effct on gaps
    g    = general intelligence
    tom  = ToM 
    psol = problem solution
    
    RESULTS:
    psol  = ability of groups to find solution to problems
    nrdiv = need for representatioinal dversity dependent on social, environemntal complexty and g
    rgap  = ability to close representational gap created by representational diversity needs, dependent on ToM
        
    """
    ecdivfac = (math.tanh(g/(10-g))**stg) * 10
    if ecdivfac < 1:
        ecdivfac = 1
    nrdiv  = scmp + ecmp / ecdivfac
    w_rgap = (math.tanh(tom/(10-tom))**stt) * 10
    rgap   = nrdiv - tom * w_rgap / 5 # max rdiv = 20, max ToM * w_rgap = 20  
    psol   = scmp + ecmp + rgap 
    psol   = 1- (psol + 20) / 60 #scaling parameter to have solutions in the 0,1 space
    return psol, nrdiv, rgap


#generate dataframe columns for solutions
solutionF = pd.DataFrame({'scomp':[],'ecomp':[], 'nrdiv':[],'rgap':[],'g':[],'alpha':[],'tom':[],'beta':[],'psol':[]})
for sc in socomp:
    for ec in ecomp:
        print(sc,ec)
        for ig in gen:
            for itom in ltom:
                for st1 in stg:
                    for st2 in stt:
                        psol2, nrdiv, rgap = fgtomF(sc,ec,st1, st2, ig,itom)
                        sol2 = pd.DataFrame({'scomp':[sc],'ecomp':[ec], 'g':[ig],'alpha':[st1],'tom':[itom], 'beta':[st2],'nrdiv':[nrdiv],'rgap':[rgap], 'psol':[psol2]})
                        solutionF = solutionF.append(sol2, sort=False)

pickle.dump (solutionF, open('toymodel.p','wb'))
solutionF.to_csv('.../toymodel.csv')

#load dataset to use for optimization 
os.chdir('/Users/jacapobaggio/Documents/A_STUDI/AAA_Work/Cognitive_Diversity/Cpx_Cog_Div_Model/DataUsed')

#os.chdir('.../Datasets/')
indiv = pd.read_csv('small_grp.csv')
state = pd.read_csv('USstates.csv')
country = pd.read_csv('countryNoSSA.csv')

os.chdir('/.../ModelSolution')
solutionF = pd.read_csv('toymodel.csv')

#assess distribution of g and ToM standardized
sns.jointplot(indiv.tom, indiv.iq)
sns.jointplot(state.tom, state.iq)
sns.jointplot(country.tom, country.iq)


#define parameters initial values and bounduary conditions
modparm = lmf.Parameters()
modparm.add_many(('ecx', 8.0, True, 0, 10),
                ('scx', 8.0, True, 0, 10),
                ('ag',1.0, True, 1, 10),
                ('bt',1.0, True, 1, 10))

#fitness function of the model, given parameter values and the vector for g, ToM and Psol
def fitting(modparm, g, tom, ps):
    parvals = modparm.valuesdict()
    scmp = parvals['scx']
    ecmp = parvals['ecx']
    stg = parvals['ag']  
    stt = parvals['bt']
    fitvalue=[0]* len(ps)
    for i in range(0,len(ps)):
        ecdivfac = (math.tanh(g[i]/(10-g[i]))**stg) * 10
        if ecdivfac < 1:
            ecdivfac = 1
        rdiv     = scmp + ecmp / ecdivfac  
        w_rgap = (math.tanh(tom[i]/(10-tom[i]))**stt) *10
        rgap     = rdiv - tom[i] * w_rgap / 5 # max rdiv = 20, max ToM * w_rgap = 20
        psol  = scmp + ecmp + rgap 
        psol = 1 - (psol + 20) / 60 #scaling parameter to have solutions in the 0,1 space
        fitvalue[i] =  abs(psol - ps[i])
    return np.prod(fitvalue)
        

#Fit model for Group, state and country dataset.
#All parameters and indep variables are scaled on a 0-10 scale 
#all dep variables are rescaled between 0 and 1 (range of psol)
#create arrays for g, tom and problem solution at the 4 levels

gg = np.array(indiv.iq)
tomg = np.array(indiv.tom)
psg = np.array(indiv.govn)

gs= np.array(state.iq)
toms = np.array(state.tom)
pss = np.array(state.govn)

gc = np.array(country.iq)
tomc = np.array(country.tom)
psc = np.array(country.govn)




#now repeat the algorithm, given stochasticity, for reps times
#storing parameter values and fitness for the reps repetitions
reps = 100
E_cpx = {}
S_cpx = {}
alpha = {}
beta = {}
fit = {}

#generating lists where to store results of repetitions
etempg = [0] * reps
stempg = [0] * reps
atempg = [0] * reps
btempg = [0] * reps
ftempg = [0] * reps

etemps = [0] * reps
stemps = [0] * reps
atemps = [0] * reps
btemps = [0] * reps
ftemps = [0] * reps

etempc = [0] * reps
stempc = [0] * reps
atempc = [0] * reps
btempc = [0] * reps
ftempc = [0] * reps



#fit data to model for the 4 level via dual_annealing (simulated annealing + fast simulated annealing)
for i in range(0,reps):
    print(i)
    print('SmallGrp')
    fit_indiv = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gg, tomg, psg))
    etempg[i] = fit_indiv.params['ecx'].value
    stempg[i] = fit_indiv.params['scx'].value
    atempg[i] = fit_indiv.params['ag'].value
    btempg[i] = fit_indiv.params['bt'].value
    ftempg[i] = fit_indiv.residual
    print('States')
    fit_state= lmf.minimize(fitting, modparm, method='dual_annealing', args=(gs, toms, pss))
    etemps[i] = fit_state.params['ecx'].value
    stemps[i] = fit_state.params['scx'].value
    atemps[i] = fit_state.params['ag'].value
    btemps[i] = fit_state.params['bt'].value
    ftemps[i] = fit_state.residual
    print('Ctr')
    fit_country = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gc, tomc, psc))
    etempc[i] =  fit_country.params['ecx'].value
    stempc[i] =  fit_country.params['scx'].value
    atempc[i] = fit_country.params['ag'].value
    btempc[i] = fit_country.params['bt'].value
    ftempc[i] = fit_country.residual 
    i+=1
#store results in dictionary, not really needed but usefful if one wants to export results

E_cpx['SmallGrp'] = etempg
S_cpx['SmallGrp'] = stempg
alpha['SmallGrp'] = atempg
beta['SmallGrp'] = btempg
fit ['SmallGrp'] = ftempg

E_cpx['State'] = etemps
S_cpx['State'] = stemps
alpha['State'] = atemps
beta['State'] = btemps
fit ['State'] = ftemps

E_cpx['Country'] = etempc
S_cpx['Country'] = stempc
alpha['Country'] = atempc
beta['Country'] = btempc
fit ['Country'] = ftempc



import pickle
pickle.dump (E_cpx, open('ecpx.p','wb')) 
pickle.dump (S_cpx, open('scpx.p','wb'))
pickle.dump (alpha, open('alpha.p','wb'))
pickle.dump (beta, open('beta.p','wb'))
pickle.dump (fit, open('fitres.p','wb'))

#Calculate and print Mean Best Value and Variance of Best value for parameters and fitness

for key in E_cpx:
    print('E_cpx,' + key)
    print('Mean Best E_cpx Value '+str(np.mean(E_cpx[key])))
    print('Var Best E_cpx Value '+str(np.var(E_cpx[key])))
for key in S_cpx:
    print('S_cpx ' + key)
    print('Mean Best S_cpx Value '+str(np.mean(S_cpx[key])))
    print('Var Best S_cpx Value '+str(np.var(S_cpx[key])))    
for key in alpha:
    print('alpha ' + key)
    print('Mean Best alpha Value '+str(np.mean(alpha[key])))
    print('Var Best alpha Value '+str(np.var(alpha[key])))
for key in beta:
    print('beta ' + key)
    print('Mean Best beta Value '+str(np.mean(beta[key])))
    print('Var Best beta Value '+str(np.var(beta[key])))
for key in fit:
    print('Fitness' + key)
    print('Mean Best fit Value '+str(np.mean(list(fit[key]))))
    print('Var Best fit Value '+str(np.var(fit[key])))

fitval ={}
for key in fit:
    fitval[key] = np.concatenate(fit[key])
    
#use mean values to build data for graphing curves to show effect of alpha and beta
#on representational diversity and representational gaps

levels = ['SmallGrp', 'State','Country','Country2']
ltom2     = list (np.arange(0, 10.1, 0.1))
gen2      = list (np.arange(0, 10.1, 0.1))

aball = {}
for key in levels:
    abcurve = pd.DataFrame({'scomp':[],'ecomp':[],'rdiv':[],'rgap':[],'g':[],'alpha':[],'tom':[],'beta':[],'psol':[]})
    #get parameters for each of the 4 levels as the mean of the best fit
    print(key)
    ec = np.mean(E_cpx[key])
    sc = np.mean(S_cpx[key])
    ag = np.mean(alpha[key])
    bt = np.mean(beta[key])
    print('sc =' + str(sc) + '; ec = ' + str(ec) + '; ag = ' + str(ag) + '; bt = ' +str(bt))
    
    for ig in gen2:
        for itom in ltom2:
            psol2, rdiv, rgap = fgtomF(sc,ec,ag, bt, ig,itom)
            sol2 = pd.DataFrame({'scomp':[sc],'ecomp':[ec], 'g':[ig],'alpha':[ag],'tom':[itom], 'beta':[bt],'rdiv':[rdiv],'rgap':[rgap], 'psol':[psol2]})
            abcurve = abcurve.append(sol2, sort=False)
    aball[key]=abcurve
    

divaball ={}
gapaball = {}
for key in aball:
    divaball[key] = pd.pivot_table(aball[key], values='rdiv', index='g', aggfunc='mean')
    gapaball[key] = pd.pivot_table(aball[key], values='rgap', index='tom',  aggfunc='mean')

dg = divaball['SmallGrp']
ds = divaball['State']
dc = divaball['Country']
dc2= divaball['Country2']

gg = gapaball['SmallGrp']
gs = gapaball['State']
gc = gapaball['Country']
gc2 = gapaball['Country2']

#calculate infleciton points from arrays. Note indices + 1 is because of issues 
#with array length for the indices, to make sure it is correctly getting the value as length(xs) =  (np.diff (np.sign (f_prime),axis=0))
xs = np.array(dg.index)
ys = dg.values
f_prime = np.gradient (ys, axis=0) # differential approximation
indices = np.where (np.diff (np.sign (f_prime),axis=0)) [0] # Find the inflection point.
inflections = xs [indices + 1]
print ('Div inflection points,Small Group', inflections) 

#calculate infleciton points from arrays
xs = np.array(ds.index)
ys = ds.values
f_prime = np.gradient (ys, axis=0) # differential approximation
indices = np.where (np.diff (np.sign (f_prime),axis=0)) [0] # Find the inflection point.
inflections = xs [indices + 1]
print ('Div inflection points,US States', inflections) 

#calculate infleciton points from arrays
xs = np.array(dc.index)
ys = dc.values
f_prime = np.gradient (ys, axis=0) # differential approximation
indices = np.where (np.diff (np.sign (f_prime),axis=0)) [0] # Find the inflection point.
inflections = xs [indices + 1] 
print ('Div inflection points,Countries', inflections) 


#Figures for Fitting
#plot fitted valus vs actual values
os.chdir('/.../Figures')
#set general context and colors for figures and graphs
hfont = {'fontname':'Times'}
sns.set(context='paper', style='white', palette='colorblind', font_scale=3)


#getting overall average g-tom complexity and steepneess  average graphs
gtomsol2= pd.pivot_table(solutionF, values='psol', index='g', columns='tom', aggfunc='mean')
escompsol2 = pd.pivot_table(solutionF, values='psol', index='scomp', columns='ecomp', aggfunc='mean')
steepdat = pd.pivot_table(solutionF, values='psol', index='alpha', columns = 'beta', aggfunc = 'mean')

#Effect on Representatinal Diversty and Representational Gaps
div1 = pd.pivot_table(solutionF, values='nrdiv', index='g', columns='alpha', aggfunc='mean')
div2 = pd.pivot_table(solutionF, values='nrdiv', index='tom', columns='beta', aggfunc='mean')
gap1 =  pd.pivot_table(solutionF, values='rgap', index='g', columns='alpha', aggfunc='mean')
gap2 =  pd.pivot_table(solutionF, values='rgap', index='tom', columns='beta', aggfunc='mean')

#usual colors for colorbar...
im = plt.imshow(gtomsol2, vmin=0, vmax=1 ,cmap='RdYlBu')
im2 = plt.imshow(div1, vmin=0, vmax=10 ,cmap='RdYlBu')

#Contour Plots 
#main paper Figure 2
f1, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, sharey=True, sharex=True, figsize=(15,15))
cbar_ax = f1.add_axes([0.99, .3, .03, .4])

subdat2 = solutionF[solutionF.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax1.set_title(r'$E_{cpx} = 10, S_{cpx} = 0$' , fontsize=24, color='black')
ax1.set_aspect('equal')
ax1.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax2.set_title(r'$E_{cpx} = 10, S_{cpx} = 5$' , fontsize=24, color='black')
ax2.set_aspect('equal')
ax2.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax3.set_title(r'$E_{cpx} = 10, S_{cpx} = 10$' , fontsize=24, color='black')
ax3.set_aspect('equal')
ax3.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax3.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax4.set_title(r'$E_{cpx} = 5, S_{cpx} = 0$' , fontsize=24, color='black')
ax4.set_aspect('equal')
ax4.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax4.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax5.set_title(r'$E_{cpx} = 5, S_{cpx} = 5$' , fontsize=24, color='black')
ax5.set_aspect('equal')
ax5.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax5.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax6.set_title(r'$E_{cpx} = 5, S_{cpx} = 10$' , fontsize=24, color='black')
ax6.set_aspect('equal')
ax6.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax6.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax7.set_title(r'$E_{cpx} = 0, S_{cpx} = 0$' , fontsize=24, color='black')
ax7.set_aspect('equal')
ax7.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax7.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax7.xaxis.set_tick_params(labelsize=20)
ax7.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax8.set_title(r'$E_{cpx} = 0, S_{cpx} = 5$' , fontsize=24, color='black')
ax8.set_aspect('equal')
ax8.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax8.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax8.xaxis.set_tick_params(labelsize=20)
ax8.yaxis.set_tick_params(labelsize=20)

subdat2 = solutionF[solutionF.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax9.set_title(r'$E_{cpx} = 0, S_{cpx} = 10$' , fontsize=24, color='black')
ax9.set_aspect('equal')
ax9.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax9.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax9.xaxis.set_tick_params(labelsize=20)
ax9.yaxis.set_tick_params(labelsize=20)

cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
f1.text(0.5,0.04, 'Social Complexity ' + r'($S_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
f1.text(0.05,0.5, 'Environmental Complexity' + r'($E_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', rotation=90, fontsize=36)
f1.text(0.5,0.07, r'$ToM$', ha='center', va='center', fontsize=36 )
f1.text(0.08,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=36)
f1.savefig('mainmodel4.pdf', bbox_inches='tight') 

#effect of g and ToM on rdiv and rhdg respectively. Second figure has also countries with SSA
#Main Paper Figure 3
fad2,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,7.5))
ax1.set_ylabel(r'$R_{div}$', fontsize = 36)
ax1.set_xlabel(r'$g$', fontsize = 36)
ax1.plot(dg, linewidth = 3, color = 'k')
ax1.plot(ds, linewidth = 3, color = 'orange')
ax1.plot(dc, linewidth = 3, color = 'r')
ax2.set_ylabel(r'$R_{hdg}$', fontsize = 36)
ax2.set_xlabel(r'$ToM$', fontsize = 36)
ax2.plot(gg, linewidth = 3, color = 'k',label='Small Groups')
ax2.plot(gs, linewidth = 3, color = 'orange', label='U.S States')
ax2.plot(gc, linewidth = 3, color = 'r', label='Countries')
ax2.axhline(y=0, xmin=0.0, xmax=10.0, linewidth=3, color='grey', linestyle='dashed')
fad2.legend()
fad2.tight_layout()
fad2.savefig('parEff.pdf')

#plotting functional forms
x = np.linspace(0,10,100)
y1 = (np.tanh(x/(10-x))**1) * max(x)
y2 = (np.tanh(x/(10-x))**2.5) * max(x)
y3 = (np.tanh(x/(10-x))**5) * max(x)
y4 = (np.tanh(x/(10-x))**7.5) * max(x)
y5 = (np.tanh(x/(10-x))**10) * max(x)

ly1 = np.where(y1<=1,1,y1)
ly2 = np.where(y2<=1,1,y2)
ly3 = np.where(y3<=1,1,y3)
ly4 = np.where(y4<=1,1,y4)
ly5 = np.where(y5<=1,1,y5)


gy1 = 5 + 5/ly3 - x * y1 / 5
gy2 = 5 + 5/ly3 - x * y2 / 5
gy3 = 5 + 5/ly3 - x * y3 / 5
gy4 = 5 + 5/ly3 - x * y4 / 5
gy5 = 5 + 5/ly3 - x * y5 / 5



y0 = 0 #point for labels

tf, (ax1, ax2) = plt.subplots(1,2, sharey=False, sharex=False, figsize=(13,5))

ax1.plot(x, 1/ly1, 'orange', linestyle='--')
ax1.plot(x, 1/ly2, 'red',    linestyle='--')
ax1.plot(x, 1/ly3, 'purple', linestyle='--')
ax1.plot(x, 1/ly4, 'blue',   linestyle='--')
ax1.plot(x, 1/ly5, 'black',  linestyle='--')
ax1.plot(x, 5/ly1, 'orange', label= r'$\alpha = 1$' )
ax1.plot(x, 5/ly2, 'red', label= r'$\alpha = 2.5$')
ax1.plot(x, 5/ly3, 'purple', label= r'$\alpha = 5$')
ax1.plot(x, 5/ly4, 'blue',label= r'$\alpha = 7.5$')
ax1.plot(x, 5/ly5, 'black', label = r'$\alpha = 10$')
ax1.plot(y0, 'black', linestyle ='--', label = r'$E_{cpx} = 1$')
ax1.plot(y0, 'black', label = r'$E_{cpx} = 5$')
ax1.plot(x, 10/ly1,'orange', linestyle='-.')
ax1.plot(x, 10/ly2,'red',    linestyle='-.')
ax1.plot(x, 10/ly3,'purple', linestyle='-.')
ax1.plot(x, 10/ly4,'blue',   linestyle='-.')
ax1.plot(x, 10/ly5,'black',  linestyle='-.', label = r'$E_{cpx} = 10$')
ax1.set_ylabel('Environmental Complexity mediated by general intelligence: ' + r'($E_{cpx} / w_{div}$)')
ax1.set_xlabel('g')

ax2.plot(x, y1, color = 'orange', label= r'$\beta = 1$')
ax2.plot(x, y2, color = 'red',    label= r'$\beta = 2.5$')
ax2.plot(x, y3, color = 'purple', label= r'$\beta = 5$')
ax2.plot(x, y4, color = 'blue',   label= r'$\beta = 7.$')
ax2.plot(x, y5, color = 'black',  label= r'$\beta = 10$')
ax2.set_ylabel(r'$w_{gap}$')
ax2.set_xlabel('ToM')
ax1.legend(loc = 'upper right', fontsize=14)
ax2.legend(loc = 'upper left', fontsize = 14)
tf.tight_layout()
tf.savefig('funcform.pdf')



tf2, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=False, sharex=False, figsize=(13,13))

ax1.plot(x, 5/ly1, 'orange',linewidth=2.0,  linestyle = '--', label= r'$\alpha = 1$' )
ax1.plot(x, 5/ly2, 'red',   linewidth=2.0,  linestyle = '--', label= r'$\alpha = 2.5$')
ax1.plot(x, 5/ly3, 'purple',linewidth=2.0, linestyle = '--', label= r'$\alpha = 5$')
ax1.plot(x, 5/ly4, 'blue',  linewidth=2.0, linestyle = '--', label= r'$\alpha = 7.5$')
ax1.plot(x, 5/ly5, 'black', linewidth=2.0, linestyle = '--', label = r'$\alpha = 10$')
ax1.set_ylabel(r'$E_{cpx} / w_{div}$')
ax1.set_xlabel('g')

ax2.plot(x, 5 + 5/ly1, linewidth=2.0, color = 'orange', linestyle='--')
ax2.plot(x, 5 + 5/ly2, linewidth=2.0, color = 'red',    linestyle='--')
ax2.plot(x, 5 + 5/ly3, linewidth=2.0, color = 'purple', linestyle='--')
ax2.plot(x, 5 + 5/ly4, linewidth=2.0, color = 'blue',   linestyle='--')
ax2.plot(x, 5 + 5/ly5, linewidth=2.0, color = 'black',  linestyle='--')
ax2.set_ylabel(r'$R_{div}$')
ax2.set_xlabel('g')

ax3.plot(x, y1, linewidth=2.0,color = 'orange', label= r'$\beta = 1$')
ax3.plot(x, y2, linewidth=2.0,color = 'red',    label= r'$\beta = 2.5$')
ax3.plot(x, y3, linewidth=2.0,color = 'purple', label= r'$\beta = 5$')
ax3.plot(x, y4, linewidth=2.0,color = 'blue',   label= r'$\beta = 7.$')
ax3.plot(x, y5, linewidth=2.0,color = 'black',  label= r'$\beta = 10$')
ax3.set_ylabel(r'$w_{gap}$')
ax3.set_xlabel('ToM')

ax4.plot(x, gy1, linewidth=2.0,color = 'orange')
ax4.plot(x, gy2, linewidth=2.0,color = 'red')
ax4.plot(x, gy3, linewidth=2.0,color = 'purple')
ax4.plot(x, gy4, linewidth=2.0,color = 'blue')
ax4.plot(x, gy5, linewidth=2.0,color = 'black')
ax4.set_ylabel(r'$R_{hdg}$')
ax4.set_xlabel('ToM')

ax1.legend(loc = 'upper right', fontsize=20)
ax3.legend(loc = 'upper left', fontsize = 20)

ax1.text(-0.05, 1.1, '(a)', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
ax2.text(-0.05, 1.1, '(b)', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
ax3.text(-0.05, 1.1, '(c)', transform=ax3.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
ax4.text(-0.05, 1.1, '(d)', transform=ax4.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

tf2.tight_layout()
tf2.savefig('funcform2.pdf')


#---------SUPPLEMENTARY FIGURES
#Supplementary Figure 1 A and 1 B
ftg, ax = plt.subplots(1,1, figsize = (15,15))
ax.contour (gtomsol2,levels = 25, linewidths=0.5, colors ='k')
cntr1 = ax.contourf(gtomsol2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax.set_xlabel ('ToM', fontsize=36)
ax.set_ylabel ('g',fontsize=36)
ax.xaxis.set_tick_params(labelsize=28)
ax.yaxis.set_tick_params(labelsize=28)
cbar = plt.colorbar(im)
cbar.set_label('Group Problem Solving Ability', fontsize = 32, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=28)
ftg.savefig('g_tom_ContourSolution.pdf')

fes, ax = plt.subplots(1,1, figsize= (15,15))
ax.contour (escompsol2,levels = 25, linewidths=0.5, colors ='k')
cntr1 = ax.contourf(escompsol2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax.set_xlabel ('Social Complexity', fontsize=36)
ax.set_ylabel ('Environmental Complexity', fontsize=36)
ax.xaxis.set_tick_params(labelsize=28)
ax.yaxis.set_tick_params(labelsize=28)
cbar = plt.colorbar(im)
cbar.set_label('Group Problem Solving Ability', fontsize = 32, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=28)
fes.savefig('Complexity_ContourSolutionl.pdf')

#supplementary figure 2
f2, axs = plt.subplots(len(socomp),len(ecomp), sharey=True, sharex=True,figsize=(30,30))
cbar_ax = f2.add_axes([0.99, .3, .03, .4])
axs.ravel()
i = 0 
j = 0
for ec in ecomp:
    for sc in socomp:
        subdat2 = solutionF[solutionF.scomp == sc]
        subdat2 = subdat2[subdat2.ecomp == ec]
        forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
        axs[i,j].set_title('EC ='+ str(ec) +','+'SC='+ str(sc), fontsize=24, color='black')
        axs[i,j].contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
        axs[i,j].contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
        #axs[ec,sc].set_xlabel('ToM', fontsize=18)
        #axs[ec,sc].set_ylabel('g', fontsize=18)
        axs[i,j].xaxis.set_tick_params(labelsize=20)
        axs[i,j].yaxis.set_tick_params(labelsize=20)
        j +=1
    i+=1
    j=0
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
fig=axs[0,0].figure
fig.text(0.5,0.07, 'Increased Social Complexity ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
fig.text(0.07,0.5, r'$\leftarrow$'+' Increased Environmental Complexity', ha='center', va='center', rotation=90, fontsize=36)
fig.text(0.5,0.1, r'$ToM$', ha='center', va='center', fontsize=42 )
fig.text(0.1,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=42)
f2.savefig('g_ToM_Cmplx_ContourSol.pdf', bbox_inches='tight') 

#-----
#Contour Plots to highlight alpha beta

solab1 = solutionF[solutionF.ecomp == 5]
solab1 = solab1[solab1.scomp == 5]

fab, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, sharey=True, sharex=True, figsize=(15,15))
cbar_ax = fab.add_axes([0.99, .3, .03, .4])

subdat2 = solab1[solab1.alpha == 1]
subdat2 = subdat2[subdat2.beta == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax1.set_title(r'$\beta = 10, \alpha = 1$' , fontsize=24, color='black')
ax1.set_aspect('equal')
ax1.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 5]
subdat2 = subdat2[subdat2.beta == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax2.set_title(r'$\beta = 10, \alpha = 5$' , fontsize=24, color='black')
ax2.set_aspect('equal')
ax2.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 10]
subdat2 = subdat2[subdat2.beta == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax3.set_title(r'$\beta = 10, \alpha = 10$' , fontsize=24, color='black')
ax3.set_aspect('equal')
ax3.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax3.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 1]
subdat2 = subdat2[subdat2.beta == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax4.set_title(r'$\beta = 5, \alpha = 1$' , fontsize=24, color='black')
ax4.set_aspect('equal')
ax4.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax4.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 5]
subdat2 = subdat2[subdat2.beta == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax5.set_title(r'$\beta = 5, \alpha = 5$' , fontsize=24, color='black')
ax5.set_aspect('equal')
ax5.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax5.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 10]
subdat2 = subdat2[subdat2.beta == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax6.set_title(r'$\beta = 5, \alpha = 10$' , fontsize=24, color='black')
ax6.set_aspect('equal')
ax6.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax6.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 1]
subdat2 = subdat2[subdat2.beta == 1]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax7.set_title(r'$\beta = 1, \alpha = 1$' , fontsize=24, color='black')
ax7.set_aspect('equal')
ax7.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax7.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax7.xaxis.set_tick_params(labelsize=20)
ax7.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 5]
subdat2 = subdat2[subdat2.beta == 1]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax8.set_title(r'$\beta = 1, \alpha = 5$' , fontsize=24, color='black')
ax8.set_aspect('equal')
ax8.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax8.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax8.xaxis.set_tick_params(labelsize=20)
ax8.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.alpha == 10]
subdat2 = subdat2[subdat2.beta == 1]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax9.set_title(r'$\beta = 1, \alpha = 10$' , fontsize=24, color='black')
ax9.set_aspect('equal')
ax9.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax9.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax9.xaxis.set_tick_params(labelsize=20)
ax9.yaxis.set_tick_params(labelsize=20)

cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
fab.text(0.5,0.04,  r'$\alpha$ ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
fab.text(0.05,0.5,  r'$\beta$ ' + r'$\rightarrow$', ha='center', va='center', rotation=90, fontsize=36)
fab.text(0.5,0.07, r'$ToM$', ha='center', va='center', fontsize=36 )
fab.text(0.08,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=36)
fab.savefig('abfigure5.pdf', bbox_inches='tight') 


#------------
#Contour Plots 
#Supplementary figure - main solutoins but for three distinct values of alpha (1,5,10) and beta (1,5,10)
#start with alpha = beta = 1

solab1 = solutionF[solutionF.alpha == 1]
solab1 = solab1[solab1.beta == 1]

f3, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, sharey=True, sharex=True, figsize=(15,15))
cbar_ax = f3.add_axes([0.99, .3, .03, .4])

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax1.set_title(r'$E_{cpx} = 10, S_{cpx} = 0$' , fontsize=24, color='black')
ax1.set_aspect('equal')
ax1.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax2.set_title(r'$E_{cpx} = 10, S_{cpx} = 5$' , fontsize=24, color='black')
ax2.set_aspect('equal')
ax2.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax3.set_title(r'$E_{cpx} = 10, S_{cpx} = 10$' , fontsize=24, color='black')
ax3.set_aspect('equal')
ax3.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax3.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax4.set_title(r'$E_{cpx} = 5, S_{cpx} = 0$' , fontsize=24, color='black')
ax4.set_aspect('equal')
ax4.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax4.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax5.set_title(r'$E_{cpx} = 5, S_{cpx} = 5$' , fontsize=24, color='black')
ax5.set_aspect('equal')
ax5.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax5.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax6.set_title(r'$E_{cpx} = 5, S_{cpx} = 10$' , fontsize=24, color='black')
ax6.set_aspect('equal')
ax6.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax6.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax7.set_title(r'$E_{cpx} = 0, S_{cpx} = 0$' , fontsize=24, color='black')
ax7.set_aspect('equal')
ax7.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax7.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax7.xaxis.set_tick_params(labelsize=20)
ax7.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax8.set_title(r'$E_{cpx} = 0, S_{cpx} = 5$' , fontsize=24, color='black')
ax8.set_aspect('equal')
ax8.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax8.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax8.xaxis.set_tick_params(labelsize=20)
ax8.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax9.set_title(r'$E_{cpx} = 0, S_{cpx} = 10$' , fontsize=24, color='black')
ax9.set_aspect('equal')
ax9.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax9.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax9.xaxis.set_tick_params(labelsize=20)
ax9.yaxis.set_tick_params(labelsize=20)

cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
f3.text(0.5,0.04, 'Social Complexity ' + r'($S_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
f3.text(0.05,0.5, 'Environmental Complexity' + r'($E_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', rotation=90, fontsize=36)
f3.text(0.5,0.07, r'$ToM$', ha='center', va='center', fontsize=36 )
f3.text(0.08,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=36)
f3.savefig('main_a1_b1.pdf', bbox_inches='tight') 

#-----------
#alpha = beta = 10
solab1 = solutionF[solutionF.alpha == 10]
solab1 = solab1[solab1.beta == 10]

f5, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, sharey=True, sharex=True, figsize=(15,15))
cbar_ax = f3.add_axes([0.99, .3, .03, .4])

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax1.set_title(r'$E_{cpx} = 10, S_{cpx} = 0$' , fontsize=24, color='black')
ax1.set_aspect('equal')
ax1.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax2.set_title(r'$E_{cpx} = 10, S_{cpx} = 5$' , fontsize=24, color='black')
ax2.set_aspect('equal')
ax2.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax3.set_title(r'$E_{cpx} = 10, S_{cpx} = 10$' , fontsize=24, color='black')
ax3.set_aspect('equal')
ax3.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax3.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax4.set_title(r'$E_{cpx} = 5, S_{cpx} = 0$' , fontsize=24, color='black')
ax4.set_aspect('equal')
ax4.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax4.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax5.set_title(r'$E_{cpx} = 5, S_{cpx} = 5$' , fontsize=24, color='black')
ax5.set_aspect('equal')
ax5.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax5.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax6.set_title(r'$E_{cpx} = 5, S_{cpx} = 10$' , fontsize=24, color='black')
ax6.set_aspect('equal')
ax6.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax6.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax7.set_title(r'$E_{cpx} = 0, S_{cpx} = 0$' , fontsize=24, color='black')
ax7.set_aspect('equal')
ax7.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax7.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax7.xaxis.set_tick_params(labelsize=20)
ax7.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax8.set_title(r'$E_{cpx} = 0, S_{cpx} = 5$' , fontsize=24, color='black')
ax8.set_aspect('equal')
ax8.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax8.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax8.xaxis.set_tick_params(labelsize=20)
ax8.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax9.set_title(r'$E_{cpx} = 0, S_{cpx} = 10$' , fontsize=24, color='black')
ax9.set_aspect('equal')
ax9.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax9.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax9.xaxis.set_tick_params(labelsize=20)
ax9.yaxis.set_tick_params(labelsize=20)

cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
f5.text(0.5,0.04, 'Social Complexity ' + r'($S_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
f5.text(0.05,0.5, 'Environmental Complexity' + r'($E_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', rotation=90, fontsize=36)
f5.text(0.5,0.07, r'$ToM$', ha='center', va='center', fontsize=36 )
f5.text(0.08,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=36)
f5.savefig('main_a10_b10.pdf', bbox_inches='tight') 

#alpha = beta = 5

solab1 = solutionF[solutionF.alpha == 5]
solab1 = solab1[solab1.beta == 5]

f4, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, sharey=True, sharex=True, figsize=(15,15))
cbar_ax = f4.add_axes([0.99, .3, .03, .4])

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax1.set_title(r'$E_{cpx} = 10, S_{cpx} = 0$' , fontsize=24, color='black')
ax1.set_aspect('equal')
ax1.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax2.set_title(r'$E_{cpx} = 10, S_{cpx} = 5$' , fontsize=24, color='black')
ax2.set_aspect('equal')
ax2.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 10]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax3.set_title(r'$E_{cpx} = 10, S_{cpx} = 10$' , fontsize=24, color='black')
ax3.set_aspect('equal')
ax3.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax3.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax4.set_title(r'$E_{cpx} = 5, S_{cpx} = 0$' , fontsize=24, color='black')
ax4.set_aspect('equal')
ax4.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax4.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax5.set_title(r'$E_{cpx} = 5, S_{cpx} = 5$' , fontsize=24, color='black')
ax5.set_aspect('equal')
ax5.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax5.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 5]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax6.set_title(r'$E_{cpx} = 5, S_{cpx} = 10$' , fontsize=24, color='black')
ax6.set_aspect('equal')
ax6.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax6.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 0]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax7.set_title(r'$E_{cpx} = 0, S_{cpx} = 0$' , fontsize=24, color='black')
ax7.set_aspect('equal')
ax7.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax7.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax7.xaxis.set_tick_params(labelsize=20)
ax7.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 5]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax8.set_title(r'$E_{cpx} = 0, S_{cpx} = 5$' , fontsize=24, color='black')
ax8.set_aspect('equal')
ax8.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax8.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax8.xaxis.set_tick_params(labelsize=20)
ax8.yaxis.set_tick_params(labelsize=20)

subdat2 = solab1[solab1.scomp == 10]
subdat2 = subdat2[subdat2.ecomp == 0]
forheat2 = pd.pivot_table(subdat2, values = 'psol', index='g', columns = 'tom', aggfunc='mean')
ax9.set_title(r'$E_{cpx} = 0, S_{cpx} = 10$' , fontsize=24, color='black')
ax9.set_aspect('equal')
ax9.contour(forheat2, levels = 25, linewidths=0.5, colors ='k')
ax9.contourf(forheat2, levels = 25, vmin=0, vmax=1, cmap='RdYlBu')
ax9.xaxis.set_tick_params(labelsize=20)
ax9.yaxis.set_tick_params(labelsize=20)

cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Group Problem Solving Ability', fontsize = 36, rotation=270, labelpad= 50)
cbar.ax.tick_params(labelsize=32)
f4.text(0.5,0.04, 'Social Complexity ' + r'($S_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', fontsize=36 )
f4.text(0.05,0.5, 'Environmental Complexity' + r'($E_{cpx}$) ' + r'$\rightarrow$', ha='center', va='center', rotation=90, fontsize=36)
f4.text(0.5,0.07, r'$ToM$', ha='center', va='center', fontsize=36 )
f4.text(0.08,0.5, r'$g$', ha='center', va='center', rotation=90, fontsize=36)
f4.savefig('main_a5_b5.pdf', bbox_inches='tight') 


#Supplementary Figure 3
fad3,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5))
ax1.set_ylabel(r'$R_{div}$', fontsize = 36)
ax1.set_xlabel(r'$g$', fontsize = 36)
ax1.plot(dg, linewidth = 3, color = 'k')
ax1.plot(ds, linewidth = 3, color = 'orange')
ax1.plot(dc, linewidth = 3, color = 'r')
ax1.plot(dc2, linewidth = 3, color = 'cyan')
ax2.set_ylabel(r'$R_{hdg}$', fontsize = 36)
ax2.set_xlabel(r'$ToM$', fontsize = 36)
ax2.plot(gg, linewidth = 3, color = 'k',label='Small Groups')
ax2.plot(gs, linewidth = 3, color = 'orange', label='U.S States')
ax2.plot(gc, linewidth = 3, color = 'r', label ='Countries without SSA')
ax2.plot(gc2, linewidth = 3, color = 'cyan', label='Countries with SSA')
ax2.axhline(y=0, xmin=0.0, xmax=10.0, linewidth=3, color='grey', linestyle='dashed')
fad3.legend()
fad3.tight_layout()
fad3.savefig('parEff_SSA.pdf')

#assess distribution similarity visually, once only for main once also with countries with SSA
#Supplementary Figure 4A and 4B
fk1, axes = plt.subplots(1,3, figsize =(30,10)) 
sns.distplot(indiv.tom, ax=axes[0], hist=False, color='black', label='Small Groups')
sns.distplot(state.tom,ax=axes[0], hist=False, color='orange',label='US State')
sns.distplot(country.tom,ax=axes[0],hist=False, color='red',label='Country')
sns.distplot(indiv.iq, ax=axes[1], hist=False, color='black', label='Small Groups')
sns.distplot(state.iq,ax=axes[1],hist=False, color='orange', label='US State')
sns.distplot(country.iq,ax=axes[1],hist=False, color='red', label='Country')
sns.distplot(indiv.govn, ax=axes[2], hist=False, color='black', label='Small Groups')
sns.distplot(state.govn,ax=axes[2],hist=False, color='orange', label='US State')
sns.distplot(country.govn,ax=axes[2],hist=False, color='red', label='Country')
axes[0].set_xlabel('ToM', color = 'black', fontsize = 36)
axes[0].set_ylabel('Density', color = 'black', fontsize = 36)
axes[1].set_xlabel('g', color = 'black', fontsize = 36)
axes[1].set_ylabel('Density', color = 'black', fontsize = 36)
axes[2].set_xlabel('Group Problem Solving Abilty', fontsize = 36)
axes[2].set_ylabel('Density', color = 'black', fontsize = 36)
fk1.tight_layout()
fk1.savefig('Kernels.pdf')


#not used figures

#some preliminary histograms to check distributionis
fh2, ((ax1,ax2,ax3),(ax4,ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15,15))
sns.distplot(indiv.tom, ax = ax1, bins=10,norm_hist=True)
ax1.set_title('Small Groups', fontsize=28, color='black')
ax1.set_ylabel ('Density',fontsize = 24, color = 'black')
ax1.set_xlabel ('ToM', fontsize = 24, color = 'black')
sns.distplot(state.tom, ax = ax2, bins=10, norm_hist=True)
ax2.set_title('U.S. State Level', fontsize=28, color='black')
ax2.set_xlabel ('ToM', fontsize = 24, color = 'black')
sns.distplot(country.tom, ax = ax3, bins=10, norm_hist=True)
ax3.set_title('Country Level', fontsize=28, color='black')
ax3.set_xlabel ('ToM', fontsize = 24, color = 'black')
sns.distplot(indiv.iq, ax = ax4, bins=10, norm_hist=True)
ax4.set_ylabel ('Density',fontsize = 24, color = 'black')
ax4.set_xlabel ('g', fontsize = 24, color = 'black')
sns.distplot(state.iq, ax = ax5, bins=10, norm_hist=True)
ax5.set_xlabel ('g', fontsize = 24, color = 'black')
sns.distplot(country.iq, ax = ax6, bins=10, norm_hist=True)
ax6.set_xlabel ('g', fontsize = 24, color = 'black')
sns.distplot(indiv.govn, ax = ax7, bins=10, norm_hist=True)
ax7.set_xlabel ('Group Problem Solving Abilty', fontsize = 24, color = 'black')
ax7.set_ylabel ('Density',fontsize = 24, color = 'black')
sns.distplot(state.govn, ax = ax8, bins=10, norm_hist=True)
ax8.set_xlabel ('Group Problem Solving Abilty', fontsize = 24, color = 'black')
sns.distplot(country.govn, ax = ax9, bins=10, norm_hist=True)
ax9.set_xlabel ('Group Problem Solving Abilty', fontsize = 24, color = 'black')
fh2.tight_layout()
fh2.savefig('ToM_g_Hist.pdf')

#more figures on rdiv g alpha and rgap tom beta
fdg, ax1 = plt.subplots(1,1, figsize= (10,10))
ax1.contour (div1,levels = 25, linewidths=0.5, colors ='k')
ax1.contourf(div1, levels = 25, vmin=0, vmax=10, cmap='RdYlBu')
ax1.set_xlabel ('g', fontsize=32)
ax1.set_ylabel (r'$\alpha$', fontsize=32)
ax1.xaxis.set_tick_params(labelsize=28)
ax1.yaxis.set_tick_params(labelsize=28)
cbar = plt.colorbar(im2)
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$R_{div}$', fontsize = 32, rotation=270, labelpad= 50)
fdg.savefig('AG_RDiv.pdf',bbox_inches='tight')

fdg, ax2 = plt.subplots(1,1, figsize= (10,10))
ax2.contour (gap2,levels = 25, linewidths=0.5, colors ='k')
ax2.contourf(gap2, levels = 25, vmin=0, vmax=10, cmap='RdYlBu')
ax2.set_xlabel ('ToM', fontsize=32)
ax2.set_ylabel (r'$\beta$', fontsize=32)
ax2.xaxis.set_tick_params(labelsize=28)
ax2.yaxis.set_tick_params(labelsize=28)
cbar = plt.colorbar(im2)
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'$R_{gap}$', fontsize = 32, rotation=270, labelpad= 50)
fdg.savefig('BT_RGap.pdf',bbox_inches='tight')



#added analysis, fit on each single data row, to assess variance in parameters not for overall, but for single groups/states/countries
#fitness function of the model, given parameter values and the vector for g, ToM and Psol
def fitting_disag(modparm, g, tom, ps):
    parvals = modparm.valuesdict()
    scmp = parvals['scx']
    ecmp = parvals['ecx']
    stg = parvals['ag']  
    stt = parvals['bt']
    ecdivfac = (math.tanh(g/(10-g))**stg) * 10
    if ecdivfac < 1:
        ecdivfac = 1
    rdiv     = scmp + ecmp / ecdivfac  
    w_rgap = (math.tanh(tom/(10-tom))**stt) *10
    rgap     = rdiv - tom * w_rgap / 5 # max rdiv = 20, max ToM * w_rgap = 20
    psol  = scmp + ecmp + rgap 
    psol = 1 - (psol + 20) / 60 #scaling parameter to have solutions in the 0,1 space
    fitvalue =  (psol - ps)**2
    return fitvalue


#now repeat the algorithm, given stochasticity, for reps times
#storing parameter values and fitness for the reps repetitions
reps = 1000
E_cpx2 = {}
S_cpx2 = {}
alpha2 = {}
beta2 = {}
fit2 = {}

etg = [0] * reps
stg = [0] * reps
atg = [0] * reps
btg = [0] * reps
ftg = [0] * reps

ets = [0] * reps
sts = [0] * reps
ats = [0] * reps
bts = [0] * reps
fts = [0] * reps

etc = [0] * reps
stc = [0] * reps
atc = [0] * reps
btc = [0] * reps
ftc = [0] * reps


#fit data to model for the 4 level via dual_annealing (simulated annealing + fast simulated annealing)
for i in range(0,reps):
    #generating lists where to store results of each data row
    etempg = [0] * len(gg)
    stempg = [0] * len(gg)
    atempg = [0] * len(gg)
    btempg = [0] * len(gg)
    ftempg = [0] * len(gg)
    
    etemps = [0] * len(gs)
    stemps = [0] * len(gs)
    atemps = [0] * len(gs)
    btemps = [0] * len(gs)
    ftemps = [0] * len(gs)

    etempc = [0] * len(gc)
    stempc = [0] * len(gc)
    atempc = [0] * len(gc)
    btempc = [0] * len(gc)
    ftempc = [0] * len(gc)
    print(i)
    
    for j in range (0, len(gg)):
        print('SmallGrp + ' + str(j))
        fit_indiv = lmf.minimize(fitting_disag, modparm, method='dual_annealing', args=(gg[j], tomg[j], psg[j]))
        etempg[j] = fit_indiv.params['ecx'].value
        stempg[j] = fit_indiv.params['scx'].value
        atempg[j] = fit_indiv.params['ag'].value
        btempg[j] = fit_indiv.params['bt'].value
        ftempg[j] = fit_indiv.residual
    for j in range(0, len(gs)):
        print('States + ' + str(j))
        fit_state= lmf.minimize(fitting_disag, modparm, method='dual_annealing', args=(gs[j], toms[j], pss[j]))
        etemps[j] = fit_state.params['ecx'].value
        stemps[j] = fit_state.params['scx'].value
        atemps[j] = fit_state.params['ag'].value
        btemps[j] = fit_state.params['bt'].value
        ftemps[j] = fit_state.residual
    for j in range(0, len(gc)):
        print('Ctr + ' + str(j))
        fit_country = lmf.minimize(fitting_disag, modparm, method='dual_annealing', args=(gc[j], tomc[j], psc[j]))
        etempc[j] =  fit_country.params['ecx'].value
        stempc[j] =  fit_country.params['scx'].value
        atempc[j] = fit_country.params['ag'].value
        btempc[j] = fit_country.params['bt'].value
        ftempc[j] = fit_country.residual 
        j=+1
    
    etg[i] = etempg
    stg[i] = stempg
    atg[i] = atempg
    btg[i] = btempg
    ftg[i] = ftempg
    
    ets[i] = etemps
    sts[i] = stemps
    ats[i] = atemps
    bts[i] = btemps
    fts[i] = ftemps
    
    etc[i] = etempc
    stc[i] = stempc
    atc[i] = atempc
    btc[i] = btempc
    ftc[i] = ftempc
    i+=1

E_cpx2['SmallGrp'] = etg
S_cpx2['SmallGrp'] = stg
alpha2['SmallGrp'] = atg
beta2['SmallGrp'] = btg
fit2 ['SmallGrp'] = ftg

E_cpx2['State'] = ets
S_cpx2['State'] = sts
alpha2['State'] = ats
beta2['State'] = bts
fit2 ['State'] = fts

E_cpx2['Country'] = etc
S_cpx2['Country'] = stc
alpha2['Country'] = atc
beta2['Country'] = btc
fit2 ['Country'] = ftc


lvl = ['SmallGrp', 'State','Country']
for key in lvl:
    pdvar = pd.DataFrame({'scomp':[],'ecomp':[],'rdiv':[],'rgap':[],'g':[],'alpha':[],'tom':[],'beta':[],'psol':[]})
    #get parameters for each of the 4 levels as the mean of the best fit
    print(key)
    ec = pd.DataFrame(E_cpx2[key]).mean().mean()
    sc = pd.DataFrame(S_cpx2[key]).mean().mean()
    ag = pd.DataFrame(alpha2[key]).mean().mean()
    bt = pd.DataFrame (beta2[key]).mean().mean()
    ft = np.mean(np.mean(pd.DataFrame(fit2[key])))
    print('Individual Row Mean')
    print('sc =' + str(sc) + '; ec = ' + str(ec) + '; ag = ' + str(ag) + '; bt = ' +str(bt) + '; fit = ' +str(ft))

lvl = ['SmallGrp', 'State','Country']
for key in lvl:
    pdvar = pd.DataFrame({'scomp':[],'ecomp':[],'rdiv':[],'rgap':[],'g':[],'alpha':[],'tom':[],'beta':[],'psol':[]})
    #get parameters for each of the 4 levels as the mean of the best fit
    print(key)
    ec = pd.DataFrame(E_cpx2[key]).mean().var()
    sc = pd.DataFrame(S_cpx2[key]).mean().var()
    ag = pd.DataFrame(alpha2[key]).mean().var()
    bt = pd.DataFrame (beta2[key]).mean().var()
    ft = np.mean(np.mean(pd.DataFrame(fit2[key])))
    print('Individual Row Variance')
    print('sc =' + str(sc) + '; ec = ' + str(ec) + '; ag = ' + str(ag) + '; bt = ' +str(bt) + '; fit = ' +str(ft))


#added analysis, fit by keeping both social and environmental complexity fixed at 1, 5 and 10.
#it is necessary, given the code, to change ecx and scx manually 
#redefine parameters initial values and bounduary conditions
modparm = lmf.Parameters()
modparm.add_many(('ecx', 1.0, False),
                ('scx', 1.0, False),
                ('ag',1.0, True, 1, 10),
                ('bt',1.0, True, 1, 10))

#now repeat the algorithm, given stochasticity, for reps times
#storing parameter values and fitness for the reps repetitions
reps = 100
E_cpx_1= {}
S_cpx_1 = {}
alpha_1 = {}
beta_1 = {}
fit_1 = {}
#generating lists where to store results of repetitions
etempg = [0] * reps
stempg = [0] * reps
atempg = [0] * reps
btempg = [0] * reps
ftempg = [0] * reps

etemps = [0] * reps
stemps = [0] * reps
atemps = [0] * reps
btemps = [0] * reps
ftemps = [0] * reps

etempc = [0] * reps
stempc = [0] * reps
atempc = [0] * reps
btempc = [0] * reps
ftempc = [0] * reps
    
for i in range(0,reps):
    print(i)
    print('SmallGrp')
    fit_indiv = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gg, tomg, psg))
    etempg[i] = fit_indiv.params['ecx'].value
    stempg[i] = fit_indiv.params['scx'].value
    atempg[i] = fit_indiv.params['ag'].value
    btempg[i] = fit_indiv.params['bt'].value
    ftempg[i] = fit_indiv.residual
    print('States')
    fit_state= lmf.minimize(fitting, modparm, method='dual_annealing', args=(gs, toms, pss))
    etemps[i] = fit_state.params['ecx'].value
    stemps[i] = fit_state.params['scx'].value
    atemps[i] = fit_state.params['ag'].value
    btemps[i] = fit_state.params['bt'].value
    ftemps[i] = fit_state.residual
    print('Ctr')
    fit_country = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gc, tomc, psc))
    etempc[i] =  fit_country.params['ecx'].value
    stempc[i] =  fit_country.params['scx'].value
    atempc[i] = fit_country.params['ag'].value
    btempc[i] = fit_country.params['bt'].value
    ftempc[i] = fit_country.residual 
    i+=1
    
E_cpx_1['SmallGrp'] = etempg
S_cpx_1['SmallGrp'] = stempg
alpha_1['SmallGrp'] = atempg
beta_1['SmallGrp'] = btempg
fit_1 ['SmallGrp'] = ftempg

E_cpx_1['State'] = etemps
S_cpx_1['State'] = stemps
alpha_1['State'] = atemps
beta_1['State'] = btemps
fit_1 ['State'] = ftemps

E_cpx_1['Country'] = etempc
S_cpx_1['Country'] = stempc
alpha_1['Country'] = atempc
beta_1['Country'] = btempc
fit_1 ['Country'] = ftempc

for key in E_cpx_1:
    print('E_cpx,' + key)
    print('Mean Best E_cpx Value '+str(np.mean(E_cpx_1[key])))
    print('Var Best E_cpx Value '+str(np.var(E_cpx_1[key])))
for key in S_cpx_1:
    print('S_cpx ' + key)
    print('Mean Best S_cpx Value '+str(np.mean(S_cpx_1[key])))
    print('Var Best S_cpx Value '+str(np.var(S_cpx_1[key])))    
for key in alpha_1:
    print('alpha ' + key)
    print('Mean Best alpha Value '+str(np.mean(alpha_1[key])))
    print('Var Best alpha Value '+str(np.var(alpha_1[key])))
for key in beta_1:
    print('beta ' + key)
    print('Mean Best beta Value '+str(np.mean(beta_1[key])))
    print('Var Best beta Value '+str(np.var(beta_1[key])))
for key in fit_1:
    print('Fitness' + key)
    print('Mean Best fit Value '+str(np.mean(list(fit_1[key]))))
    print('Var Best fit Value '+str(np.var(fit_1[key])))

fitval_1 ={}
for key in fit_1:
    fitval_1[key] = np.concatenate(fit_1[key])

###

#added analysis, fit by keeping both social and environmental complexity fixed at 1, 5 and 10.
#it is necessary, given the code, to change ecx and scx manually 
#redefine parameters initial values and bounduary conditions
modparm = lmf.Parameters()
modparm.add_many(('ecx', 5.0, False),
                ('scx', 5.0, False),
                ('ag',1.0, True, 1, 10),
                ('bt',1.0, True, 1, 10))

#now repeat the algorithm, given stochasticity, for reps times
#storing parameter values and fitness for the reps repetitions
reps = 100
E_cpx_5= {}
S_cpx_5 = {}
alpha_5 = {}
beta_5 = {}
fit_5 = {}
#generating lists where to store results of repetitions
etempg = [0] * reps
stempg = [0] * reps
atempg = [0] * reps
btempg = [0] * reps
ftempg = [0] * reps

etemps = [0] * reps
stemps = [0] * reps
atemps = [0] * reps
btemps = [0] * reps
ftemps = [0] * reps

etempc = [0] * reps
stempc = [0] * reps
atempc = [0] * reps
btempc = [0] * reps
ftempc = [0] * reps

#fit data to model for the 3 level via dual_annealing (simulated annealing + fast simulated annealing)
for i in range(0,reps):
    print(i)
    print('SmallGrp')
    fit_indiv = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gg, tomg, psg))
    etempg[i] = fit_indiv.params['ecx'].value
    stempg[i] = fit_indiv.params['scx'].value
    atempg[i] = fit_indiv.params['ag'].value
    btempg[i] = fit_indiv.params['bt'].value
    ftempg[i] = fit_indiv.residual
    print('States')
    fit_state= lmf.minimize(fitting, modparm, method='dual_annealing', args=(gs, toms, pss))
    etemps[i] = fit_state.params['ecx'].value
    stemps[i] = fit_state.params['scx'].value
    atemps[i] = fit_state.params['ag'].value
    btemps[i] = fit_state.params['bt'].value
    ftemps[i] = fit_state.residual
    print('Ctr')
    fit_country = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gc, tomc, psc))
    etempc[i] =  fit_country.params['ecx'].value
    stempc[i] =  fit_country.params['scx'].value
    atempc[i] = fit_country.params['ag'].value
    btempc[i] = fit_country.params['bt'].value
    ftempc[i] = fit_country.residual 
    i+=1

E_cpx_5['SmallGrp'] = etempg
S_cpx_5['SmallGrp'] = stempg
alpha_5['SmallGrp'] = atempg
beta_5['SmallGrp'] = btempg
fit_5['SmallGrp'] = ftempg

E_cpx_5['State'] = etemps
S_cpx_5['State'] = stemps
alpha_5['State'] = atemps
beta_5['State'] = btemps
fit_5 ['State'] = ftemps

E_cpx_5['Country'] = etempc
S_cpx_5['Country'] = stempc
alpha_5['Country'] = atempc
beta_5['Country'] = btempc
fit_5 ['Country'] = ftempc

for key in E_cpx_5:
    print('E_cpx,' + key)
    print('Mean Best E_cpx Value '+str(np.mean(E_cpx_5[key])))
    print('Var Best E_cpx Value '+str(np.var(E_cpx_5[key])))
for key in S_cpx_5:
    print('S_cpx ' + key)
    print('Mean Best S_cpx Value '+str(np.mean(S_cpx_5[key])))
    print('Var Best S_cpx Value '+str(np.var(S_cpx_5[key])))    
for key in alpha_5:
    print('alpha ' + key)
    print('Mean Best alpha Value '+str(np.mean(alpha_5[key])))
    print('Var Best alpha Value '+str(np.var(alpha_5[key])))
for key in beta_5:
    print('beta ' + key)
    print('Mean Best beta Value '+str(np.mean(beta_5[key])))
    print('Var Best beta Value '+str(np.var(beta_5[key])))
for key in fit_5:
    print('Fitness' + key)
    print('Mean Best fit Value '+str(np.mean(list(fit_5[key]))))
    print('Var Best fit Value '+str(np.var(fit_5[key])))

fitval_5 ={}
for key in fit_5:
    fitval_5[key] = np.concatenate(fit_5[key])

##
modparm.add_many(('ecx', 10.0, False),
                ('scx', 10.0, False),
                ('ag',1.0, True, 1, 10),
                ('bt',1.0, True, 1, 10))

#now repeat the algorithm, given stochasticity, for reps times
#storing parameter values and fitness for the reps repetitions
reps = 100
E_cpx_10= {}
S_cpx_10 = {}
alpha_10 = {}
beta_10 = {}
fit_10 = {}
#generating lists where to store results of repetitions
etempg = [0] * reps
stempg = [0] * reps
atempg = [0] * reps
btempg = [0] * reps
ftempg = [0] * reps

etemps = [0] * reps
stemps = [0] * reps
atemps = [0] * reps
btemps = [0] * reps
ftemps = [0] * reps

etempc = [0] * reps
stempc = [0] * reps
atempc = [0] * reps
btempc = [0] * reps
ftempc = [0] * reps
    
for i in range(0,reps):
    print(i)
    print('SmallGrp')
    fit_indiv = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gg, tomg, psg))
    etempg[i] = fit_indiv.params['ecx'].value
    stempg[i] = fit_indiv.params['scx'].value
    atempg[i] = fit_indiv.params['ag'].value
    btempg[i] = fit_indiv.params['bt'].value
    ftempg[i] = fit_indiv.residual
           
    print('States')
    fit_state= lmf.minimize(fitting, modparm, method='dual_annealing', args=(gs, toms, pss))
    etemps[i] = fit_state.params['ecx'].value
    stemps[i] = fit_state.params['scx'].value
    atemps[i] = fit_state.params['ag'].value
    btemps[i] = fit_state.params['bt'].value
    ftemps[i] = fit_state.residual
        
    print('Ctr')
    fit_country = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gc, tomc, psc))
    etempc[i] =  fit_country.params['ecx'].value
    stempc[i] =  fit_country.params['scx'].value
    atempc[i] = fit_country.params['ag'].value
    btempc[i] = fit_country.params['bt'].value
    ftempc[i] = fit_country.residual 
        
    i+=1
    
E_cpx_10['SmallGrp'] = etempg
S_cpx_10['SmallGrp'] = stempg
alpha_10['SmallGrp'] = atempg
beta_10['SmallGrp'] = btempg
fit_10 ['SmallGrp'] = ftempg

E_cpx_10['State'] = etemps
S_cpx_10['State'] = stemps
alpha_10['State'] = atemps
beta_10['State'] = btemps
fit_10 ['State'] = ftemps

E_cpx_10['Country'] = etempc
S_cpx_10['Country'] = stempc
alpha_10['Country'] = atempc
beta_10['Country'] = btempc
fit_10 ['Country'] = ftempc

for key in E_cpx_10:
    print('E_cpx,' + key)
    print('Mean Best E_cpx Value '+str(np.mean(E_cpx_10[key])))
    print('Var Best E_cpx Value '+str(np.var(E_cpx_10[key])))
for key in S_cpx_10:
    print('S_cpx ' + key)
    print('Mean Best S_cpx Value '+str(np.mean(S_cpx_10[key])))
    print('Var Best S_cpx Value '+str(np.var(S_cpx_10[key])))    
for key in alpha_10:
    print('alpha ' + key)
    print('Mean Best alpha Value '+str(np.mean(alpha_10[key])))
    print('Var Best alpha Value '+str(np.var(alpha_10[key])))
for key in beta_10:
    print('beta ' + key)
    print('Mean Best beta Value '+str(np.mean(beta_10[key])))
    print('Var Best beta Value '+str(np.var(beta_10[key])))
for key in fit_10:
    print('Fitness' + key)
    print('Mean Best fit Value '+str(np.mean(list(fit_10[key]))))
    print('Var Best fit Value '+str(np.var(fit_10[key])))

fitval_10 ={}
for key in fit_10:
    fitval_10[key] = np.concatenate(fit_10[key])

#added analysis - using Nelder Mead max likelhood
for i in range(0,10):
    print(i)
    print('SmallGrp')
    fit_indiv = lmf.minimize(fitting, modparm, method='nelder', args=(gg, tomg, psg))
    etempg[i] = fit_indiv.params['ecx'].value
    stempg[i] = fit_indiv.params['scx'].value
    atempg[i] = fit_indiv.params['ag'].value
    btempg[i] = fit_indiv.params['bt'].value
    ftempg[i] = fit_indiv.residual
    print('States')
    fit_state= lmf.minimize(fitting, modparm, method='nelder', args=(gs, toms, pss))
    etemps[i] = fit_state.params['ecx'].value
    stemps[i] = fit_state.params['scx'].value
    atemps[i] = fit_state.params['ag'].value
    btemps[i] = fit_state.params['bt'].value
    ftemps[i] = fit_state.residual
    print('Ctr')
    fit_country = lmf.minimize(fitting, modparm, method='nelder', args=(gc, tomc, psc))
    etempc[i] =  fit_country.params['ecx'].value
    stempc[i] =  fit_country.params['scx'].value
    atempc[i] = fit_country.params['ag'].value
    btempc[i] = fit_country.params['bt'].value
    ftempc[i] = fit_country.residual 
    i+=1


#fitness function of the model, given parameter values and the vector for g, ToM and Psol 
def fitting2(modparm, g, tom, ps):
    parvals = modparm.valuesdict()
    scmp = parvals['scx']
    ecmp = parvals['ecx']
    stg = parvals['ag']  
    stt = parvals['bt']
    ecdivfac = (np.tanh(g/(10-g))**stg) * 10
    ecd2 = np.where (ecdivfac < 1, 1,ecdivfac) #constrain ecd to >=1
    rdiv     = scmp + ecmp / ecd2  
    w_rgap = (np.tanh(tom/(10-tom))**stt) *10
    rgap     = rdiv - tom * w_rgap / 5 # max rdiv = 20, max ToM * w_rgap = 20
    psol  = scmp + ecmp + rgap 
    psol = 1 - (psol + 20) / 60 #scaling parameter to have solutions in the 0,1 space
    fitvalue =  abs(psol - ps)
    return fitvalue



modparm = lmf.Parameters()
modparm.add_many(('ecx',1.0, False, 0, 10),
                 ('scx', 1.0, False, 0, 10),
                 ('ag',5.0, True, 1, 10),
                 ('bt',5.0, True, 1, 10))

print('SmallGrp')
fit_indiv = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gg, tomg, psg))
print(lmf.fit_report(fit_indiv))
print('Residuals ' + str(np.mean(fit_indiv.residual)))
           
print('US States')
fit_state= lmf.minimize(fitting, modparm, method='dual_annealing', args=(gs, toms, pss))
print(lmf.fit_report(fit_state))
print('Residuals ' + str(np.mean(fit_state.residual)))

print('Countries')
fit_country = lmf.minimize(fitting, modparm, method='dual_annealing', args=(gc, tomc, psc))
print(lmf.fit_report(fit_country))
print('Residuals ' + str(np.mean(fit_country.residual)))