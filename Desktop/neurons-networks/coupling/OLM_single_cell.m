function Y = HH(t, X)

C=1.3;
gna = 30.; vna = 90.; 
gk  = 23.; vk = -100.;
ga = 16.; va = -90.;
gh = 12.; vh = -32.9;
gl = 0.05; vl = -70.; 
Iapp = -4.70;

V = X(1);
m = X(2);
h = X(3);
n = X(4);
a = X(5);
b = X(6);
r = X(7);

Y(1) = (1/C)*( -(gna*m.^3*h*(V-vna) + gk*n.^4*(V-vk) + ga*a*b*(V-va) ...
        + gh*r*(V-vh) + gl*(V-vl) ) + Iapp );
Y(2) = am(V)*(1-m) - bm(V)*m;
Y(3) = ah(V)*(1-h) - bh(V)*h;
Y(4) = an(V)*(1-n) - bn(V)*n;
Y(5) = (ainf(V)-a)./taua(V);
Y(6) = (binf(V)-b)./taub(V);
Y(7) = (rinf(V)-r)./taur(V);

Y = Y';

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