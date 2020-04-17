function [feature_out,reco] = extract_features2(nn, x,sigma,mu)
    nn.testing = 1;
    dim_data=size(x);
     
    xx=x;
    x = normalize(x, mu, sigma);

     min_data=min(x);
     max_data=max(x);
    xxx=x;
     x=rescale(x);
     %x=(x+1)/2;
    
   % [x, mu, sigma] = zscore(x);
    
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    features=nn.a{3};
    dime=size(features);
    feature_out=features(:,2:dime(2));
    
    reco=nn.a{5};
    
     reco=reco*2-1;
%   



dataout = reco - min_data;

dataout = (dataout/range(dataout(:))).*(max_data-min_data);
dataout = dataout + min_data;

    
    
    reco=bsxfun(@times,dataout,sigma);
    reco=bsxfun(@plus,reco,mu);
    
    
    a=0;
    

    
    
end
