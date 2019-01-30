function Y =vl_euclideanloss(X, c, dzdy)

assert(numel(X) == numel(c));

d = size(X);

assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    Y =  1 / 2 * sum(subsref((X - c) .^ 2, substruct('()', {':'})));
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y =  dzdy * (X - c)/numel(X);
    
end

end