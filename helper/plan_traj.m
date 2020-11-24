%%
function [v,w,a,alpha,t_d] = plan_traj(x,y,v_max,dt,rampduration)

R = (x^2 + y^2)/(2*abs(x)); % radius of circular trajectory
r = sqrt(x^2 + y^2);
theta = atan2(abs(y),abs(x)); % target angle
phi = atan2(r*sin(theta), R - r*cos(theta)); % angle subtended by the arc (start -> target)
d = R*phi; % trajectory length
t_d = d/v_max; % movement duration

v = v_max*ones(round(t_d/dt),1); % linear velocity
v(1 : rampduration) = linspace(0,v_max,rampduration); % ramp up
v(end+1 : end+rampduration) = linspace(v_max,0,rampduration); % ramp down
w = sign(x)*(v/R); % angular velocity

a = diff(v)/dt;
alpha = diff(w)/dt;

end