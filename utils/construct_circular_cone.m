function reg_window=construct_circular_cone(sz)
x = linspace(-sqrt(2)/2,sqrt(2)/2,sz(1));
y = linspace(-sqrt(2)/2,sqrt(2)/2,sz(2));
[xx,yy] = ndgrid(x,y);
reg_window =1 - sqrt((xx.^2 + yy.^2));

end