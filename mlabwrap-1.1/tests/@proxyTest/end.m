function res=end(X,K,N)
disp(sprintf('end called: %d %d\n', K, N));
res=length(X.data); %FIXME how do you do I *call* end???
end