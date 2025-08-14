-- Enhanced Black-Scholes Database Schema with ML Optimization
-- Create database and tables for MySQL

-- Create database
CREATE DATABASE IF NOT EXISTS `Evan_dev_db` 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_0900_ai_ci;

USE `Evan_dev_db`;

-- Drop existing tables if needed (careful in production!)
-- DROP TABLE IF EXISTS `BlackScholesOutputs`;
-- DROP TABLE IF EXISTS `MLDataCache`;
-- DROP TABLE IF EXISTS `BlackScholesInputs`;

-- Table 1: Black-Scholes Input Parameters
-- Stores all input parameters for each calculation
CREATE TABLE IF NOT EXISTS `BlackScholesInputs` (
  `CalculationId` INT NOT NULL AUTO_INCREMENT,
  
  -- Core Black-Scholes parameters
  `StockPrice` DECIMAL(18,9) NOT NULL,
  `StrikePrice` DECIMAL(18,9) NOT NULL,
  `InterestRate` DECIMAL(18,9) NOT NULL,
  `Volatility` DECIMAL(18,9) NOT NULL,
  `TimeToExpiry` DECIMAL(18,9) NOT NULL,
  
  -- Purchase prices for P&L calculation
  `PurchaseCallPrice` DECIMAL(18,9) NULL,
  `PurchasePutPrice` DECIMAL(18,9) NULL,
  
  -- Grid parameters for heatmap
  `MinSpot` DECIMAL(18,9) NOT NULL,
  `MaxSpot` DECIMAL(18,9) NOT NULL,
  `MinVol` DECIMAL(18,9) NOT NULL,
  `MaxVol` DECIMAL(18,9) NOT NULL,
  `GridNSpot` INT NOT NULL DEFAULT 50,
  `GridNVol` INT NOT NULL DEFAULT 50,
  
  -- Metadata for ML tracking
  `CreatedAt` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `SessionId` VARCHAR(100) NULL,
  `UserId` VARCHAR(100) NULL,
  
  PRIMARY KEY (`CalculationId`),
  
  -- Indexes for efficient querying
  INDEX `idx_stock_price` (`StockPrice`),
  INDEX `idx_strike_price` (`StrikePrice`),
  INDEX `idx_volatility` (`Volatility`),
  INDEX `idx_time_to_expiry` (`TimeToExpiry`),
  INDEX `idx_created_at` (`CreatedAt`),
  INDEX `idx_session_id` (`SessionId`),
  INDEX `idx_user_id` (`UserId`),
  
  -- Composite indexes for ML queries
  INDEX `idx_bs_inputs_params` (`StockPrice`, `StrikePrice`, `Volatility`, `TimeToExpiry`),
  INDEX `idx_bs_inputs_session` (`SessionId`, `CreatedAt`),
  INDEX `idx_bs_inputs_user` (`UserId`, `CreatedAt`)
  
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table 2: Black-Scholes Output Grid
-- Stores calculated option prices for each grid point
CREATE TABLE IF NOT EXISTS `BlackScholesOutputs` (
  `CalculationOutputId` INT NOT NULL AUTO_INCREMENT,
  
  -- Grid coordinates
  `VolatilityShock` DECIMAL(18,9) NOT NULL,
  `StockPriceShock` DECIMAL(18,9) NOT NULL,
  
  -- Calculated option price at this point
  `OptionPrice` DECIMAL(18,9) NOT NULL,
  
  -- Option type: 1 = Call, 0 = Put
  `IsCall` TINYINT(1) NOT NULL,
  
  -- Reference to input parameters
  `CalculationId` INT NOT NULL,
  
  PRIMARY KEY (`CalculationOutputId`),
  
  -- Foreign key constraint
  CONSTRAINT `FK_BlackScholesInput_BlackScholesOutput_CalculationId`
    FOREIGN KEY (`CalculationId`)
    REFERENCES `BlackScholesInputs` (`CalculationId`)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  
  -- Indexes for efficient retrieval
  INDEX `FK_BlackScholesInput_BlackScholesOutput_CalculationId_idx` (`CalculationId`),
  INDEX `idx_volatility_shock` (`VolatilityShock`),
  INDEX `idx_stock_price_shock` (`StockPriceShock`),
  INDEX `idx_is_call` (`IsCall`),
  
  -- Composite indexes for grid queries
  INDEX `idx_bs_outputs_grid` (`CalculationId`, `IsCall`, `VolatilityShock`, `StockPriceShock`),
  INDEX `idx_bs_outputs_option_type` (`IsCall`, `CalculationId`)
  
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table 3: ML Data Cache (Optional)
-- Caches preprocessed features for ML training
CREATE TABLE IF NOT EXISTS `MLDataCache` (
  `CacheId` INT NOT NULL AUTO_INCREMENT,
  `CalculationId` INT NOT NULL,
  
  -- Feature vector (stored as JSON)
  `Features` VARCHAR(5000) NOT NULL,
  
  -- Labels
  `CallPrice` FLOAT NOT NULL,
  `PutPrice` FLOAT NOT NULL,
  
  -- Greeks (optional)
  `CallDelta` FLOAT NULL,
  `PutDelta` FLOAT NULL,
  `Gamma` FLOAT NULL,
  `Vega` FLOAT NULL,
  `CallTheta` FLOAT NULL,
  `PutTheta` FLOAT NULL,
  `CallRho` FLOAT NULL,
  `PutRho` FLOAT NULL,
  
  -- Metadata
  `CreatedAt` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  PRIMARY KEY (`CacheId`),
  
  -- Foreign key
  CONSTRAINT `FK_MLCache_BlackScholesInputs_CalculationId`
    FOREIGN KEY (`CalculationId`)
    REFERENCES `BlackScholesInputs` (`CalculationId`)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  
  -- Indexes
  INDEX `idx_ml_cache_calc` (`CalculationId`),
  INDEX `idx_ml_cache_created` (`CreatedAt`)
  
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Create views for common queries

-- View: Recent calculations summary
CREATE OR REPLACE VIEW `v_recent_calculations` AS
SELECT 
    i.CalculationId,
    i.StockPrice,
    i.StrikePrice,
    i.Volatility,
    i.TimeToExpiry,
    i.InterestRate,
    i.CreatedAt,
    i.UserId,
    COUNT(DISTINCT o.CalculationOutputId) as OutputCount
FROM `BlackScholesInputs` i
LEFT JOIN `BlackScholesOutputs` o ON i.CalculationId = o.CalculationId
WHERE i.CreatedAt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY i.CalculationId
ORDER BY i.CreatedAt DESC;

-- View: User activity summary
CREATE OR REPLACE VIEW `v_user_activity` AS
SELECT 
    UserId,
    COUNT(DISTINCT CalculationId) as TotalCalculations,
    AVG(StockPrice) as AvgStockPrice,
    AVG(Volatility) as AvgVolatility,
    MIN(CreatedAt) as FirstCalculation,
    MAX(CreatedAt) as LastCalculation
FROM `BlackScholesInputs`
WHERE UserId IS NOT NULL
GROUP BY UserId;

-- Stored procedures for common operations

DELIMITER $$

-- Procedure: Get calculation with outputs
CREATE PROCEDURE `sp_get_calculation`(
    IN p_calculation_id INT
)
BEGIN
    -- Get input parameters
    SELECT * FROM `BlackScholesInputs` 
    WHERE CalculationId = p_calculation_id;
    
    -- Get call outputs
    SELECT 
        VolatilityShock,
        StockPriceShock,
        OptionPrice
    FROM `BlackScholesOutputs`
    WHERE CalculationId = p_calculation_id 
    AND IsCall = 1
    ORDER BY VolatilityShock, StockPriceShock;
    
    -- Get put outputs
    SELECT 
        VolatilityShock,
        StockPriceShock,
        OptionPrice
    FROM `BlackScholesOutputs`
    WHERE CalculationId = p_calculation_id 
    AND IsCall = 0
    ORDER BY VolatilityShock, StockPriceShock;
END$$

-- Procedure: Clean up old data
CREATE PROCEDURE `sp_cleanup_old_data`(
    IN p_days_to_keep INT
)
BEGIN
    DECLARE v_cutoff_date DATETIME;
    DECLARE v_deleted_count INT;
    
    SET v_cutoff_date = DATE_SUB(NOW(), INTERVAL p_days_to_keep DAY);
    
    -- Delete old calculations (cascade will handle outputs)
    DELETE FROM `BlackScholesInputs`
    WHERE CreatedAt < v_cutoff_date;
    
    SET v_deleted_count = ROW_COUNT();
    
    SELECT v_deleted_count as DeletedCalculations;
END$$

-- Procedure: Get ML training data
CREATE PROCEDURE `sp_get_ml_training_data`(
    IN p_limit INT,
    IN p_user_id VARCHAR(100)
)
BEGIN
    SELECT 
        i.CalculationId,
        i.StockPrice,
        i.StrikePrice,
        i.InterestRate,
        i.Volatility,
        i.TimeToExpiry,
        i.StockPrice / i.StrikePrice as Moneyness,
        AVG(CASE WHEN o.IsCall = 1 THEN o.OptionPrice END) as AvgCallPrice,
        AVG(CASE WHEN o.IsCall = 0 THEN o.OptionPrice END) as AvgPutPrice,
        i.CreatedAt
    FROM `BlackScholesInputs` i
    INNER JOIN `BlackScholesOutputs` o ON i.CalculationId = o.CalculationId
    WHERE (p_user_id IS NULL OR i.UserId = p_user_id)
    GROUP BY i.CalculationId
    ORDER BY i.CreatedAt DESC
    LIMIT p_limit;
END$$

DELIMITER ;

-- Create indexes for performance optimization
-- Additional indexes can be added based on query patterns

-- Performance indexes for common WHERE clauses
CREATE INDEX `idx_perf_moneyness` ON `BlackScholesInputs` 
    ((StockPrice / StrikePrice));

-- Partitioning for large datasets (optional, requires MySQL 5.7+)
-- ALTER TABLE `BlackScholesOutputs` 
-- PARTITION BY RANGE (CalculationId) (
--     PARTITION p0 VALUES LESS THAN (1000000),
--     PARTITION p1 VALUES LESS THAN (2000000),
--     PARTITION p2 VALUES LESS THAN (3000000),
--     PARTITION p3 VALUES LESS THAN MAXVALUE
-- );

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON `Evan_dev_db`.* TO 'ml_user'@'%';
-- GRANT EXECUTE ON `Evan_dev_db`.* TO 'ml_user'@'%';

-- Initial data check
SELECT 
    'Database setup complete' as Status,
    DATABASE() as CurrentDatabase,
    COUNT(*) as TableCount
FROM information_schema.tables 
WHERE table_schema = 'Evan_dev_db';