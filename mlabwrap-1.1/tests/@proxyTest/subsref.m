function X2=subsref(X,sel)
% a hack
if (any(strcmp(sel.type, {'()', '{}'})) && ...
    length(sel.subs)==1 && ...
    strcmp('char',class(sel.subs{1})) &&...
    ~strcmp(sel.subs{1}, ':')), % yuckyuckyuck ain't matlab nice?
  X2=['you ', sel.type, '-indexed with the string <<', sel.subs{1}, '>>'];
else
  switch sel.type,
    % just a hack to have something to test {} indexing which has no direct
    % counterpart in python; we treat it as () indexing, except that the on
    % setting we convert from string and on getting to string
   case {'{}'}
    sel.type = '()';
    X2=num2str(subsref(X.data,sel));
   otherwise
    X2=subsref(X.data, sel);
  end
end
end