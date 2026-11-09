"""Gradient elution as a contact ladder with a time-varying contact strength.
Fractional position xi(t): dxi/dt = 1/(t0*(1+k)), ln k = a - S*(phi - 0.5),
phi = pump organic fraction at time t - tD - xi*t0 (the composition that has reached the analyte)."""
import numpy as np
def make_program(times, phis):
    times=np.asarray(times,float); phis=np.asarray(phis,float)
    return lambda t: np.interp(t, times, phis, left=phis[0], right=phis[-1])
def retention(prog, t0, tD, a, S, dt=0.005, tmax=30.0):
    """All arguments broadcast together; returns retention times (nan if not eluted by tmax)."""
    b=np.broadcast_arrays(np.asarray(t0,float),np.asarray(tD,float),np.asarray(a,float),np.asarray(S,float))
    shape=b[0].shape; t0,tD,a,S=[x.ravel().copy() for x in b]
    xi=np.zeros(a.size); tr=np.full(a.size,np.nan); t=0.0; idx=np.arange(a.size)
    while t<tmax and idx.size:
        phi=prog(t - tD[idx] - xi[idx]*t0[idx])
        v=1.0/(t0[idx]*(1.0+np.exp(np.clip(a[idx]-S[idx]*(phi-0.5),-30,30))))
        xn=xi[idx]+v*dt
        done=xn>=1.0
        if done.any():
            j=idx[done]; tr[j]=t+(1.0-xi[j])/(xn[done]-xi[j])*dt
        xi[idx]=xn; idx=idx[~done]; t+=dt
    return tr.reshape(shape)
