xb_DE_time = X120_DE_time(1:120000);
xi_DE_time = X171_DE_time(1:120000);%105
xo_DE_time = X199_DE_time(1:120000);%130
xii_DE_time = X107_DE_time(1:120000);
%A1 = random('Normal',0,1,1,2);%[-0.6156,0.7481]
A1 = [0.4 0.6];
xbi2_DE_time = [xb_DE_time xi_DE_time]*A1';
%A2 = random('Normal',0,1,1,2);%[-0.1924,0.8886]
A2 = [0.5 0.5];
xbo2_DE_time = [xb_DE_time xo_DE_time]*A2';
%A3 = random('Normal',0,1,1,2);%[-0.7648,-1.402]
A3 = [0.7 0.3];
xio2_DE_time = [xii_DE_time xo_DE_time]*A3';
save('0.007-Ball+0.014-Inner.mat','xbi2_DE_time');
save('0.007-Ball+0.014-Outer.mat','xbo2_DE_time');
save('0.007-Inner+0.014-Outer.mat','xio2_DE_time');
xbi2_DE_time_1 = tanh(xbi2_DE_time);
xbo2_DE_time_1 = tanh(xbo2_DE_time);
xio2_DE_time_1 = tanh(xio2_DE_time);


xbi2_DE_time_2 = 2./(1+exp(-xbi2_DE_time))-1;
xbo2_DE_time_2 = 2./(1+exp(-xbo2_DE_time))-1;
xio2_DE_time_2 = 2./(1+exp(-xio2_DE_time))-1;
save('0.007-Ball+0.014-Inner_1.mat','xbi2_DE_time_1');
save('0.007-Ball+0.014-Inner_2.mat','xbi2_DE_time_2');
save('0.007-Ball+0.014-Outer_1.mat','xbo2_DE_time_1');
save('0.007-Ball+0.014-Outer_2.mat','xbo2_DE_time_2');
save('0.007-Inner+0.014-Outer_1.mat','xio2_DE_time_1');
save('0.007-Inner+0.014-Outer_2.mat','xio2_DE_time_2');

