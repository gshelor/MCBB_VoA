/// stan model for calculating the defensive rating for the MCBB VoA
// intended to represent the number of points a given team will score against the hypothetical average D1 team

data {
  int<lower=0> N;
  vector[N] adjdef_ppg;
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
  real beta_def_ppp;
  real beta_def_assists;
  real beta_def_trueshooting;
  real beta_def_oppblocks;
  real beta_def_oppsteals;
  real beta_def_fgpct;
  real beta_def_2ptfgpct;
  real beta_def_3ptfgpct;
  real beta_def_ftpct;
  real beta_def_rebounds;
  real beta_def_turnovers;
  real<lower=0> sigma;
}

model {
  // Priors (Matching your PyMC mu and sigma)
  b0 ~ normal(50, 5);
  beta_def_ppp ~ normal(2, 10);
  beta_def_assists ~ normal(1, 10);
  beta_def_trueshooting ~ normal(2.5, 10);
  beta_def_oppblocks ~ normal(-2, 5);
  beta_def_oppsteals ~ normal(-2, 5);
  beta_def_fgpct ~ normal(2, 20);
  beta_def_2ptfgpct ~ normal(1, 15);
  beta_def_3ptfgpct ~ normal(1, 15);
  beta_def_ftpct ~ normal(1, 15);
  beta_def_rebounds ~ normal(2, 15);
  beta_def_turnovers ~ normal(-1, 5);
  sigma ~ inv_gamma(0, 10); // Half-normal is enforced by <lower=0> in parameters

  // Likelihood
  adjdef_ppg ~ normal(
    b0 + 
    beta_def_ppp * ppp + 
    beta_def_assists * assists + 
    beta_def_trueshooting * trueshooting + 
    beta_def_oppblocks * oppblocks + 
    beta_def_oppsteals * oppsteals + 
    beta_def_fgpct * fgpct + 
    beta_def_2ptfgpct * twoptfgpct + 
    beta_def_3ptfgpct * threeptfgpct + 
    beta_def_ftpct * ftpct + 
    beta_def_rebounds * rebounds + 
    beta_def_turnovers * turnovers, 
    sigma
  );
}