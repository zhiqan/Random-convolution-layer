xb_DE_time = X120_DE_time(1:120000); %120
xi_DE_time = X107_DE_time(1:120000); %107
xo_DE_time = X132_DE_time(1:120000); %132
%A1 = random('Normal',0,1,1,2);%[-0.6156,0.7481]
A1 = [0.4 0.6];
xbi_DE_time = [xb_DE_time xi_DE_time]*A1';
%A2 = random('Normal',0,1,1,2);%[-0.1924,0.8886]
A2 = [0.5 0.5];
xbo_DE_time = [xb_DE_time xo_DE_time]*A2';
%A3 = random('Normal',0,1,1,2);%[-0.7648,-1.402]
A3 = [0.7 0.3];
xio_DE_time = [xi_DE_time xo_DE_time]*A3';

A4 = [0.3 0.4 0.3];
xbio_DE_time = [xb_DE_time xi_DE_time xo_DE_time]*A4';
xbio_DE_time_1 = tanh(xbi_DE_time);
save('0.007-Ball+Inner+Outer_1.mat','xbio_DE_time_1');

% save('0.007-Ball+Inner.mat','xbi_DE_time');
% save('0.007-Ball+Outer.mat','xbo_DE_time');
% save('0.007-Inner+Outer.mat','xio_DE_time');
% xbi_DE_time_1 = tanh(xbi_DE_time);
% xbo_DE_time_1 = tanh(xbo_DE_time);
% xio_DE_time_1 = tanh(xio_DE_time);
% 
% xbi_DE_time_2 = 2./(1+exp(-xbi_DE_time))-1;
% xbo_DE_time_2 = 2./(1+exp(-xbo_DE_time))-1;
% xio_DE_time_2 = 2./(1+exp(-xio_DE_time))-1;
% save('0.007-Ball+Inner_1.mat','xbi_DE_time_1');
% save('0.007-Ball+Inner_2.mat','xbi_DE_time_2');
% save('0.007-Ball+Outer_1.mat','xbo_DE_time_1');
% save('0.007-Ball+Outer_2.mat','xbo_DE_time_2');
% save('0.007-Inner+Outer_1.mat','xio_DE_time_1');
% save('0.007-Inner+Outer_2.mat','xio_DE_time_2');
% 
