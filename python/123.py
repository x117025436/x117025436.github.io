import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 6 , 100)
x1 = [t* 0.5 * np.pi for t in x]
y=np.sin(x1)
plt.figure()
plt.subplot(1,4,1)
plt.plot(x1,y)
plt.title(r'$f(t)=sin(\omega t)u(t), \omega = \frac{4}{8} \pi$') 
#plt.show()

x2=np.linspace(2, 10 , 100)
x3=[t* 0.5 * np.pi for t in x2]
y2=np.sin(x3)
plt.subplot(1,4,2)
plt.plot(x3,y2)
plt.title(r'$f(t)=sin{\omega (t-t0)}u(t), \omega = \frac{4}{8}\pi,t0=2 $') 
#plt.show()

x4 = np.linspace(2, 6 , 100)
x5= [t* 0.5 * np.pi for t in x4]
y3=np.sin(x5)
plt.subplot(1,4,3)
plt.plot(x5,y3)
plt.title(r'$f(t)=sin(\omega t)u(t-t0), \omega = \frac{4}{8}\pi,t0=2 $') 
#plt.show()

x6= np.linspace(2, 8 , 100)
x7= [t* 0.5 * np.pi for t in x6]
y4=np.sin(x7)
plt.subplot(1,4,4)
plt.plot(x7,y4)
plt.title(r'$f(t)=sin{\omega （t-t0) } u(t-t0), \omega = \frac{4}{8}\pi,t0=2 $') 
plt.show()










