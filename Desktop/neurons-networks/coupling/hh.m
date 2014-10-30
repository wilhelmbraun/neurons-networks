function Y = HH(t, X, G, Iapp)

N = size(G,1);

% olm-cell parameters
C=1.3;
gna = 30.; vna = 90.; 
gk  = 23.; vk = -100.;
ga = 16.; va = -90.;
gh = 12.; vh = -32.9;
gl = 0.05; vl = -70.; 

% synaptic connection parameters
vgaba = -80.;
taurise=0.2;
taudecay=20.;

V = zeros(1,N);
m = zeros(1,N);
h = zeros(1,N);
n = zeros(1,N);
a = zeros(1,N);
b = zeros(1,N);
r = zeros(1,N);
s = zeros(1,N);

dimHH=8;

for i=1:N
    k=(i-1)*dimHH;
    V(i) = X(k+1);
    m(i) = X(k+2);
    h(i) = X(k+3);
    n(i) = X(k+4);
    a(i) = X(k+5);
    b(i) = X(k+6);
    r(i) = X(k+7);
    s(i) = X(k+8);
end

%S=heaviside(t-synConnectTime).*s;
I_app=Iapp(t);
Y=zeros(1,dimHH*N);

for j=1:N
    k=(j-1)*dimHH;
    V_=V(j);
    m_=m(j);
    h_=h(j);
    n_=n(j);
    a_=a(j);
    b_=b(j);
    r_=r(j);
    s_=s(j);
    
    Y(k+1) = (1/C)*( -(gna*m_.^3*h_*(V_-vna) + gk*n_.^4*(V_-vk) + ga*a_.*b_.*(V_-va) ...
            + gh*r_.*(V_-vh) + gl*(V_-vl)  ...
            + (G(i,:)*s')*(V_-vgaba)) + I_app(j) );
        
    Y(k+2) = am(V_)*(1-m_) - bm(V_)*m_;
    Y(k+3) = ah(V_)*(1-h_) - bh(V_)*h_;
    Y(k+4) = an(V_)*(1-n_) - bn(V_)*n_;
    Y(k+5) = (ainf(V_)-a_)./taua(V_);
    Y(k+6) = (binf(V_)-b_)./taub(V_);
    Y(k+7) = (rinf(V_)-r_)./taur(V_);
    Y(k+8) = p(V_)*(1-s_)./taurise - s_./taudecay;
        
end

Y = Y';

% gating functions
function y = p(V)
   y = (1.+tanh(V./4))/2.;
end

function y = am(V)
y = -0.1*(V+38.)./(exp(-(V+38.)/10.)-1.);
end

function y =  bm(V)
y = 4.*exp(-(V+65.)/18.);
end

function y =  ah(V)
y = 0.07*exp(-(V+63.)/20.);
end

function y =  bh(V)
y =  1./(exp(-(V+33.)/10.)+1);
end

function y = an(V)
y =  0.018*(V-25.)/(1-exp(-(V-25.)/25.));
end

function y = bn(V)
y = 0.0036*(V-35.)/(exp((V-35.)/12.)-1.);
end

function y = ainf(V)
y = 1./(1.+exp(-(V+14.)/16.6));
end

function y = taua(V)
y=5.;
end

function y = binf(V)
y = 1./(1+exp((V+71.)/7.3));
end

function y = taub(V)
y=1./(0.000009/exp((V-26.)/18.5)+0.014/(0.2+exp(-(V+70.)/11.)));
end

function y = rinf(V)
y = 1./(1+exp((V+84.)/10.2));
end

function y = taur(V)
y = 1./(exp(-14.59-0.086*V) + exp(-1.87 + 0.0701*V));
end

end