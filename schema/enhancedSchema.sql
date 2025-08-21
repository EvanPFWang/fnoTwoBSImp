CREATE DATABASE IF NOT EXISTS blackscholes;
USE blackscholes;

CREATE TABLE IF NOT EXISTS calculations (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  s DOUBLE NOT NULL,
  k DOUBLE NOT NULL,
  r DOUBLE NOT NULL,
  q DOUBLE NOT NULL,
  t DOUBLE NOT NULL,
  sigma DOUBLE,
  option_type ENUM('call','put') NOT NULL,
  price DOUBLE,
  delta DOUBLE,
  gamma DOUBLE,
  theta DOUBLE,
  vega DOUBLE,
  rho DOUBLE,
  implied_vol DOUBLE,
  market_price DOUBLE
);
