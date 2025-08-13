-- Create DB and tables (MySQL)
CREATE DATABASE IF NOT EXISTS `Evan_dev_db` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `Evan_dev_db`;

CREATE TABLE IF NOT EXISTS `BlackScholesInputs` (
  `CalculationId` INT NOT NULL AUTO_INCREMENT,
  `StockPrice` DECIMAL(18,9) NOT NULL,
  `StrikePrice` DECIMAL(18,9) NOT NULL,
  `InterestRate` DECIMAL(18,9) NOT NULL,
  `Volatility` DECIMAL(18,9) NOT NULL,
  `TimeToExpiry` DECIMAL(18,9) NOT NULL,
  `PurchaseCallPrice` DECIMAL(18,9) NULL,
  `PurchasePutPrice` DECIMAL(18,9) NULL,
  `MinSpot` DECIMAL(18,9) NOT NULL,
  `MaxSpot` DECIMAL(18,9) NOT NULL,
  `MinVol` DECIMAL(18,9) NOT NULL,
  `MaxVol` DECIMAL(18,9) NOT NULL,
  `GridNSpot` INT NOT NULL DEFAULT 50,
  `GridNVol` INT NOT NULL DEFAULT 50,
  PRIMARY KEY (`CalculationId`)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS `BlackScholesOutputs` (
  `CalculationOutputId` INT NOT NULL AUTO_INCREMENT,
  `VolatilityShock` DECIMAL(18,9) NOT NULL,
  `StockPriceShock` DECIMAL(18,9) NOT NULL,
  `OptionPrice` DECIMAL(18,9) NOT NULL,
  `IsCall` TINYINT(1) NOT NULL,
  `CalculationId` INT NOT NULL,
  PRIMARY KEY (`CalculationOutputId`),
  KEY `FK_BlackScholesInput_BlackScholesOutput_CalculationId_idx` (`CalculationId`),
  CONSTRAINT `FK_BlackScholesInput_BlackScholesOutput_CalculationId`
    FOREIGN KEY (`CalculationId`)
    REFERENCES `BlackScholesInputs` (`CalculationId`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION
) ENGINE=InnoDB;
