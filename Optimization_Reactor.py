import numpy as np
from gekko import GEKKO
from scipy.optimize import differential_evolution

def objective(params):
    x2_0, T1 = params
    
    m = GEKKO(remote=False)
    
    nt = 501 
    m.time = np.linspace(0,4,nt)
    
    c1 = (1-x2_0)*0.35/0.65
    c3 = (1-x2_0)*0.3/0.65
    #c1+c3=98.8 kmol/h

    # Variables
    x1 = m.Var(value=c1)
    x2 = m.Var(value=x2_0) 
    x3 = m.Var(value=c3)
    x4 = m.Var(value=0)
    x5 = m.Var(value=0)
    x6 = m.Var(value=0)
    x7 = m.Var(value=0)
    T = m.Const(value=T1)
    
    A1 = m.Const(value=4340)
    A2 = m.Const(value=5873)
    A3 = m.Const(value=46782)
    A4 = m.Const(value=57304)
    E1 = m.Const(value=19113)
    E2 = m.Const(value=27190)
    E3 = m.Const(value=49195)
    E4 = m.Const(value=55852)
    R = m.Const(value=8.31)
    
    k1 = m.Intermediate(A1*m.exp(-E1/(R*T))) 
    k2 = m.Intermediate(A2*m.exp(-E2/(R*T)))
    k3 = m.Intermediate(A3*m.exp(-E3/(R*T)))
    k4 = m.Intermediate(A4*m.exp(-E4/(R*T)))
    
    r1 = m.Intermediate(k1*(x1)*(x2)*(x3))
    r2 = m.Intermediate(k2*(x1)*(x2)*(x3))
    r3 = m.Intermediate(k3*(x4)*(x5))
    r4 = m.Intermediate(k4*(x4)*(x5))
    
    m.Equation(x1.dt()== -r1-r2)
    m.Equation(x2.dt()== -r1-r2)
    m.Equation(x3.dt()== -r1-r2)
    m.Equation(x4.dt()== 3*(r1-r3-r4))
    m.Equation(x5.dt()== 3*(r2-r3-r4))
    m.Equation(x6.dt()== 3*r3)
    m.Equation(x7.dt()== 3*r4)
    
    m.options.IMODE = 4
    m.solve(disp=False)

    #convert to kmol
    F_total=(1/(c1+c3))*98.8
    #mass flowrate
    F_mass=(c1*96+x2_0*2+c3*126)*F_total
    #product total molar flowrate
    F_molar=F_mass/(x4[-1]*321.5+x5[-1]*144.9+x6[-1]*18+x7[-1]*44)

    #Volume
    pc=1500 #kgm-3, catalyst density
    E=0.488 #catalyst voidage 
    X = x4[-1] + x5[-1] + x6[-1] + x7[-1] #conversion
    V = F_mass * (X / (pc * (1-E) * (3*(r1[50] + r2[50] - 2*r3[50] - 2*r4[50]))))
    D=pow(((4/(3.14159*3))*V),1/3) #diameter
    H=D*3 #height, diameter-to-height-ratio is 1/3

    #15 years_life_time
    Imsk= 2171.6 #2020
    Imsd= 1695.1 #2010
    Fm= 3.75 #construction factor
    Fp= 1.5#pressure factor
    Fi= 1.5 #installation factor in Malaysia

    Kr= (Imsk/Imsd)*7775.3*(D**1.066)*(H**0.82)*(2.18+Fm*Fp)*Fi#installation cost, RM (not RM million ya, only RM)
    Krt = Kr*15/3 #15 years of annual cost
    #CO2 cost = -RM 0.18/kg

    #cost
    C_H2=x2_0*F_total*8.47*2*8520 #RM 8.47/kg, 2 g/mol, F_total=kmol/h, 355 days

    #profit (use mass fraction)
    #P=r*V (mol/s)
    P_spk = 3*(r1[50]-r3[50]-r4[50])*V*9.84*355*86400 #RM 9.84/kg, 321.5 g/mol, F_total=kmol/h, 355 days
    P_gasoline = 3*(r2[50]-r3[50]-r4[50])*V*4.26*355*86400 #RM 4.26/kg, 144.9 g/mol, F_total=kmol/h, 355 days
    
    #other raw materials since constant(regardless of how many H2), thus not included in calculation
    Profit = (P_spk+P_gasoline)*15-Kr-Krt-C_H2*15

    return -Profit  # maximize x4 at final time,-x4.value[-1]

bounds = [(0.33, 0.5), (436, 473)]  # bounds for x2_0 and T1
result = differential_evolution(objective, bounds,disp=True)

print(f"Optimal x2_0: {result.x[0]:.4f}")  
print(f"Optimal T1: {result.x[1]:.2f} K")
print(f"Maximum Profit: {-result.fun:.4f}")