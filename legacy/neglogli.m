function [loss, dL] = neglogli(prm, prm_index, k_base_conv, h_base_conv, sps, dtSp)
Itot = k_base_conv * prm(prm_index{1}) + prm(prm_index{2}) + h_base_conv * prm(prm_index{3});
rr = exp(Itot);
loss = -sum(Itot(sps > 0)) + sum(rr) * dtSp;

% gradient
dLdk = -sum(k_base_conv(sps > 0, :)) + (rr' * k_base_conv) * dtSp;
dLdc = -sum(sps > 0) + sum(rr) * dtSp;
dLdh = -sum(h_base_conv(sps > 0, :)) + (rr' * h_base_conv) * dtSp;

dL = [dLdk, dLdc, dLdh]';
