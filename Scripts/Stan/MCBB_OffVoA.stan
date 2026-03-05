/// stan model for calculating the offensive rating for the MCBB VoA
// intended to represent the number of points a given team will score against the hypothetical average D1 team

data {
  int<lower=0> N;
  vector[N] adjoff_ppg;
  vector[N] ppp;
  vector[N] assists;
  vector[N] trueshooting;
  vector[N] oppblocks;
  vector[N] oppsteals;
  vector[N] fgpct;
  vector[N] twoptfgpct;
  vector[N] threeptfgpct;
  vector[N] ftpct;
  vector[N] rebounds;
  vector[N] turnovers;
}

parameters {
  real b0;
  real beta_off_ppp;
  real beta_off_assists;
  real beta_off_trueshooting;
  real beta_off_oppblocks;
  real beta_off_oppsteals;
  real beta_off_fgpct;
  real beta_off_2ptfgpct;
  real beta_off_3ptfgpct;
  real beta_off_ftpct;
  real beta_off_rebounds;
  real beta_off_turnovers;
  real<lower=0> sigma;
}

model {
  // Priors (Matching your PyMC mu and sigma)
  b0 ~ normal(50, 5);
  beta_off_ppp ~ normal(2, 10);
  beta_off_assists ~ normal(1, 10);
  beta_off_trueshooting ~ normal(2.5, 10);
  beta_off_oppblocks ~ normal(-2, 5);
  beta_off_oppsteals ~ normal(-2, 5);
  beta_off_fgpct ~ normal(2, 20);
  beta_off_2ptfgpct ~ normal(1, 15);
  beta_off_3ptfgpct ~ normal(1, 15);
  beta_off_ftpct ~ normal(1, 15);
  beta_off_rebounds ~ normal(2, 15);
  beta_off_turnovers ~ normal(-1, 5);
  sigma ~ inv_gamma(0, 10); // Half-normal is enforced by <lower=0> in parameters

  // Likelihood
  adjoff_ppg ~ normal(
    b0 + 
    beta_off_ppp * ppp + 
    beta_off_assists * assists + 
    beta_off_trueshooting * trueshooting + 
    beta_off_oppblocks * oppblocks + 
    beta_off_oppsteals * oppsteals + 
    beta_off_fgpct * fgpct + 
    beta_off_2ptfgpct * twoptfgpct + 
    beta_off_3ptfgpct * threeptfgpct + 
    beta_off_ftpct * ftpct + 
    beta_off_rebounds * rebounds + 
    beta_off_turnovers * turnovers, 
    sigma
  );
}