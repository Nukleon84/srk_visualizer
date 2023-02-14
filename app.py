import streamlit as st
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import fsolve

class SRK:
    def __init__(self,TC,PC, AC, T,P):
        self.TC=TC
        self.PC=PC
        self.AC=AC
        self.R=8.31446261815324 #J/mol/K
        self.T=T
        self.P=P
        self.TR=T/TC
        self.PR=P/PC
        m=(0.48508 + 1.55171*AC - 0.15613*AC*AC)        
        self.alpha=(1 + m * (1 - np.sqrt(self.TR)))**2

        self.update()

        #print()
        #print("TR=",self.TR, "PR=", self.PR)
        
    
    def update(self):
        self.a      = self.alpha*0.427480*(self.R*self.TC)**2/self.PC
        self.b      = 0.086649*self.R*self.TC/self.PC
        self.rho0   = 1.0/self.b
        self.rhomc  = 0.2599/self.b
        return        
      
    
    def pressure(self,v):
        R=self.R
        T=self.T               
        a=self.a
        b=self.b
        return R*T/(v-b) - a/(v*(v + b))
        
    def molar_volumes(self):
        A=self.a
        B=self.b
        T=self.T
        P=self.P
        R=self.R

        p= -(P*B**2+R*T*B-A)/3/P - (R*T/3/P)**2
        q = -(R*T/3/P)**3 - R*T*(P*B**2+ R*T*B-A)/6/P**2-A*B/2/P
        diskr=q**2+p**3

        if(diskr<0):
            r = np.sign(q)*np.sqrt(np.abs(p))
            cosphi=q/r**3
            phi=np.arccos(cosphi)
            x1=-2*r*np.cos(phi/3)
            x2= 2*r*np.cos((np.pi-phi)/3)
            x3= 2*r*np.cos((np.pi+phi)/3)
            vv = np.max([x1, x2, x3])+R*T/3/P
            vl = np.min([x1, x2, x3])+R*T/3/P            
            return (vl,vv)
        else:
            h1=-q+np.sqrt(diskr)
            h2=-q-np.sqrt(diskr)
            h3=np.sign(h1)
            h4=np.sign(h2)
            v=h3*np.abs(h1)**(1/3) +h4*np.abs(h2)**(1/3)
            #vv=R*T/P
            vl=v+R*T/3/P
            vv=vl            
            return (vl,vv)
    
    def lnphi(self, Z):
        R=self.R
        T=self.T
        P=self.P
        A= self.a*P/((R*T)**2)
        B=self.b*P/(R*T)                                         
        q=A/(B)       
        return Z-1 - np.log(Z-B) - q*np.log((Z+ B)/Z)

    def fugacities(self,vl,vv):
        R=self.R
        T=self.T
        P=self.P                           
        
        zl = P*vl/(R*T)
        zv = P*vv/(R*T)     
        lnphi_l=self.lnphi(zl)
        lnphi_v=self.lnphi(zv)
        return (lnphi_l, lnphi_v)
    
    

def plot_srk(TC,PC,AC, T,P):
     
    sys = SRK(TC,PC,AC,T,P) 
    pvap_sys = SRK(TC,PC,AC,T,P)
 
    def eos_ideal_p(vbar):
        v=vbar
        p=sys.R*T/v
        return p
    def solve_pvap(p, sys):
        sys.P=p[0]
        sys.update()
        vl,vv=sys.molar_volumes()
        (lnphi_l, lnphi_v)= sys.fugacities(vl,vv)
        return lnphi_l-lnphi_v

    (vl,vv)=sys.molar_volumes()
    (lnphi_l, lnphi_v)= sys.fugacities(vl,vv)

    v_eos= np.linspace(sys.b+1e-6, vv+1e-4, 2000)
    rho_eos= np.linspace(1e-2, (sys.rho0-1), 1000)
    P_eos=np.array(list(map(sys.pressure, v_eos)))
    P_eos_ideal=np.array(list(map(eos_ideal_p, v_eos)))

    P_eos_rho=np.array(list(map(sys.pressure,1/rho_eos)))
    P_eos_ideal_rho=np.array(list(map(eos_ideal_p,1/rho_eos)))

   
    if(show_vaporpressure)    :
        p0= max(1e3, pvap_sys.pressure(1/(pvap_sys.rhomc))) 
        pvap=[p0]
        x, results, error, mesg=fsolve(solve_pvap, pvap, args=pvap_sys, full_output=True)    
        pvap=x[0]
              
    col1,col2,col3=st.columns(3)
    col1.metric("Fugacity Coefficient Liquid", round(np.exp(lnphi_l),4))
    col2.metric("Fugacity Coefficient Vapour", round(np.exp(lnphi_v),4))
    if(show_vaporpressure)    :
        col3.metric("Vapor Pressure", f"{round(pvap/1e5,2)} bar")
    
    fig=go.Figure()
    fig.add_scatter(    
            x=v_eos*1e6,
            y=P_eos/1e5,
            name="Pressure (SRK)",
            line=dict(color="blue"),
    
    )
    if show_ideal:
        fig.add_scatter(    
                x=v_eos*1e6,
                y=P_eos_ideal/1e5,
                name="Pressure (ideal)",
                line=dict(color="red"),              
        )
   
    fig.add_scatter(    
                x=[vv*1e6,vl*1e6],
                y=[P/1e5,P/1e5],
                name="Roots",
                line=dict(color="green"),
                marker=dict(size=12),
                mode="markers"
                
        )

    
    fig.add_hline(y=P/1e5,line_dash="dash", line_color="green",annotation_text="P_sys")
    if show_vaporpressure:
        fig.add_hline(y=pvap/1e5,line_dash="dash", line_color="maroon",annotation_text="P_vap")
    
    fig.update_layout(        
        width=600,
        height=600,
        title="Pressure vs. Molar Volume",
        xaxis_title="Molar Volume [cm³/mol]",
        yaxis_title="Pressure [bar]",
        legend_title="Legend",
        font=dict(            
            size=12,            
        )
    )

    fig.update_xaxes(showgrid=True, zeroline=False, zerolinecolor="#444", range=(0, vv*1e6+100),zerolinewidth=2,showline=True, titlefont = dict(size=20))
    fig.update_yaxes(showgrid=True, zeroline=False, zerolinecolor="#444",range=(0, 100), zerolinewidth=2,showline=True,titlefont = dict(size=20))

    # Second Plot
    fig2=go.Figure() 
    fig2.add_scatter(    
            x=rho_eos/1e6,
            y=P_eos_rho/1e5,
            name="Pressure (SRK)",
            line=dict(color="blue"),
           
           
    )
    if show_ideal:
        fig2.add_scatter(    
                x=rho_eos/1e6,
                y=P_eos_ideal_rho/1e5,
                name="Pressure (ideal)",
                line=dict(color="red"),
              
        )
    fig2.add_hline(y=P/1e5,line_dash="dash", line_color="green",annotation_text="P_sys")
    
    fig2.add_vline(x=sys.rho0/1e6,line_dash="dot", line_color="cyan",annotation_text="rho0")
    fig2.add_vline(x=sys.rhomc/1e6,line_dash="dot", line_color="cyan",annotation_text="rho_mc")
  
    fig2.add_scatter(    
                x=[1/vv/1e6,1/vl/1e6],
                y=[P/1e5,P/1e5],
                name="Roots",
                line=dict(color="green"),
                marker=dict(size=12),
                mode="markers"
                
        )
    fig2.update_layout(        
        width=600,
        height=600,
        title="Pressure vs. Density",
        xaxis_title="Density [mol/cm³]",
        yaxis_title="Pressure [bar]",
        legend_title="Legend",
        font=dict(            
            size=12,            
        )
    )
    
  
    fig2.update_xaxes(showgrid=True, zeroline=False, zerolinecolor="#444", range=(0, (sys.rho0+2e3)/1e6),zerolinewidth=2,showline=True, titlefont = dict(size=20))
    fig2.update_yaxes(showgrid=True, zeroline=False, zerolinecolor="#444",range=(0, 100), zerolinewidth=2,showline=True,titlefont = dict(size=20))

    return (fig, fig2)

st.set_page_config(layout="wide") 

st.title("Interactive Visualizer for the Soave-Redlich-Kwong Equation-of-State")


with st.sidebar:
    
    st.header("Configuration")
    st.subheader("Properties")
    TC=st.slider("Critical Temperature [K]",10.0,1000.0, 647.096) #Propane 369.8
    PC=st.slider("Critical Pressure [bar]",10.0,300.0, 220.64) #Propane 42.48
    AC=st.slider("Acentric Factor [-]",0.0,1.0,0.344) #Propane 0.152

    st.subheader("System")
    T=st.slider("System temperature [K]", 0.0,1.1*TC,373.15)#0.7*TC) #Propane 311.11
    P=st.slider("System pressure [bar]", 1.0,100.0,12.977)

    st.subheader("Options")
    #st.checkbox("Show Pseudo-Roots")
    show_ideal=st.checkbox("Show Ideal Gas")
    show_vaporpressure=st.checkbox("Calculate Vapor Pressure")

fig1,fig2 =plot_srk(TC,PC*1e5,AC,T,P*1e5)

col1,col2= st.columns(2)
col1.plotly_chart(fig1, theme="streamlit", use_container_width=True)
col2.plotly_chart(fig2, theme="streamlit", use_container_width=True)