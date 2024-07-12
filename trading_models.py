# Standard library imports
from typing import Tuple, List, Union

# Third-party library imports
import yfinance as yf
import pandas as pd
import arch
import numpy as np
import statsmodels.tsa as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


class BaseModel:
    """
    Base class for fetching historical stock data for multiple stocks.
    """
    def __init__(self, tickers: Union[str, List[str], np.array], period: str = "ytd", interval: str = "1d") -> None:
        self.tickers = np.array([tickers]) if isinstance(tickers, str) else np.array(tickers)
        self.period = period
        self.interval = interval
        self.hist = self.get_price_history()

    def get_price_history(self) -> pd.DataFrame:
        """
        Fetch the historical stock price data for multiple stocks.
        
        Returns:
        - hist (pd.DataFrame): Historical stock price data.
        """
        hist = yf.download(self.tickers.tolist(), period=self.period, interval=self.interval)
        # If only one ticker is provided, yfinance doesn't return a multi-level column index
        if isinstance(hist.columns, pd.Index):
            hist.columns = pd.MultiIndex.from_product([hist.columns, self.tickers])
        return hist

class DataPreprocessor(BaseModel):
    """
    Handle data preprocessing operations for multiple stocks.
    """
    def __init__(self, tickers: Union[str, List[str], np.array], period: str = "ytd", interval: str = "1d") -> None:
        super().__init__(tickers, period, interval)
        self.ccr_matrix = self._calculate_ccr()
    
    def _calculate_ccr(self) -> pd.DataFrame:
        """
        Calculate the continuously compounded returns (CCR) for multiple stocks.
        
        Returns:
        - ccr (pd.DataFrame): Continuously compounded returns for each stock.
        """
        close_prices = self.hist['Close']
        ccr = np.log(close_prices / close_prices.shift(1)).dropna()
        return ccr


class ARMAModel(DataPreprocessor):
    """
    Handle ARMA model operations.
    """
    def __init__(self, tickers: Union[str, List[str], np.array], stock: str = None, period: str = "ytd", interval: str = "1d") -> None:
        super().__init__(tickers, period, interval)
        if stock:
            if isinstance(tickers, (np.array, list)):
                if stock not in self.tickers:
                    raise ValueError(f"Stock {stock} not in the list of tickers.")
                self.stock = stock
                self.ccr = self.ccr_matrix[stock]
            elif isinstance(tickers, str):
                if stock != tickers:
                    raise ValueError(f"Specified stock {stock} does not match the single ticker {tickers}")
                self.stock = stock
                self.ccr = self.ccr_matrix[stock]
        else:
            if isinstance(tickers, str):
                self.stock = tickers
                self.ccr = self.ccr_matrix[tickers]
            elif len(tickers) == 1:
                self.stock = tickers[0]
                self.ccr = self.ccr_matrix[tickers[0]]
            else:
                raise ValueError("Must specify a stock when multiple tickers are provided")

        self.model = None

    def arma_order(self) -> tuple[int, int, int]:
        """
        Stepwise ARMA selection using AIC

        Returns:
        - best_order (tuple): AIC informed best order (p, d, q)
        """
        # Determine the maximum feasible AR terms to add
        max_p = 1
        aic1, aic2 = np.inf, np.inf - 1

        while aic2 < aic1:
            aic1 = aic2
            # Fit AR model
            ar_model = sm.AutoReg(self.ccr, lags=max_p).fit()
            # Calculate AIC
            aic2 = ar_model.aic
            # Increment parameters
            max_p += 1
        max_p -= 1

        # Determine the maximum feasible MA terms to add
        max_q = 1
        aic1, aic2 = np.inf, np.inf - 1

        while aic2 < aic1:
            aic1 = aic2
            # Fit MA model
            ma_model = sm.ARIMA(self.ccr, order=(0, 0, max_q)).fit()
            # Calculate AIC
            aic2 = ma_model.aic
            # Increment parameters
            max_q += 1
        max_q -= 1

        lowest_aic = np.inf
        possible_orders = [(p, q) for p in range(1, max_p + 1) for q in range(1, max_q + 1)]

        for p, q in possible_orders:
            try:
                model = sm.ARIMA(self.ccr, order=(p, 0, q)).fit()
                
                # Check residual autocorrelation using Ljung-Box test
                lb_test = acorr_ljungbox(model.resid, lags=min(10, len(model.resid)-1))
                if lb_test[1][0] < 0.05:  # Significance Level
                    continue  # Skip this order if autocorrelation is significant
                
                if model.aic < lowest_aic:
                    lowest_aic = model.aic
                    best_order = p, 0, q
            except Exception as e:
                print(f"Error fitting ARIMA({p}, 0, {q}): {e}")
                continue
        
        return best_order
    
    def fit_arma(self, order: tuple[int,int,int] = None, window: tuple[int,int] = (0, None)) -> sm.ARIMA:
        """
        Fit an ARMA model to the time series data.
        
        If no order is provided, this method identifies the best-fitting ARMA model
        for the time series data stored in `self.ccr` using stepwise selection
        that minimizes the Akaike Information Criterion (AIC).

        Parameters:
        - order (tuple[int,int,int], optional): The order of the ARMA model (p,d,q). If None, the best order will be determined
            using stepwise selection. Default is None.
        - window (tuple[int, int], optional): The start and end indices of the data window to use for fitting.
            Default is (0, None), which uses all available data.

        Returns:
        - The fitted ARIMA model (sm.ARIMA).

        Raises:
        ValueError
            If the provided order is invalid or if the window indices are out of range.
        """
        # Define parameters for model order and data
        if not order:
            order = self.arma_order()
        start, end = window
        if not end:
            end = len(self.ccr)
        
        # Return the model corresponding to the given window and order
        model = sm.ARIMA(self.ccr[start:end], order=order).fit()
        self.model = model
        return model

    def arma_forecast(self, model: sm.ARIMA, steps_ahead: int = 1) -> np.array:
        """
        Generate forecasts using the provided ARIMA model.

        Parameters:
        - model (sm.ARIMA): Fitted ARIMA model from statsmodels. The model should
        already be trained on historical data and ready for forecasting.

        Returns:
        - np.array: Array of forecasted values based on the ARIMA model.

        Raises:
        - ValueError: If the provided `model` is not an instance of `sm.ARIMA`.
        """
        if model is None:
            if self.model is None:
                raise ValueError("No model has been fit. Call fit_arma() first or provide a model.")
            model = self.model
        
        if not isinstance(model, sm.ARIMA):
            raise ValueError("The `model` parameter must be an instance of sm.ARIMA.") 
        
        # Generate forecasts using the ARIMA model
        forecast_values = model.forecast(steps=steps_ahead)

        return forecast_values


class VARModel(DataPreprocessor):
    """
    Handle Vector Autoregression model operations for multiple stocks.
    """
    def __init__(self, tickers: Union[str, List[str], np.array], period: str = "ytd", interval: str = "1d") -> None:
        if isinstance(tickers, str) or (isinstance(tickers, (list, np.array)) and len(tickers) < 2):
            raise ValueError("VARModel requires at least two stocks. Use ARMAModel for single stock analysis.")
        
        super().__init__(tickers, period, interval)
        
        if self.ccr_matrix.shape[1] < 2:
            raise ValueError(f"Insufficient data: VARModel requires at least two stocks with valid data, but only {self.ccr_matrix.shape[1]} stock(s) have data.")
        
        self.model = None

    def fit_var(self, order: int = 1, window: tuple[int,int] = (0, None)) -> sm.api.VAR:
        """
        Fit a VAR model to the time series data.

        Parameters:
        - order (int, optional): The order of the VAR model.
        - window (tuple[int, int], optional): The start and end indices of the data window to use for fitting.
            Default is (0, None), which uses all available data.

        Returns:
        - The fitted VAR model (sm.api.VAR).

        Raises:
        - ValueError: If the provided order is invalid or if the window indices are out of range.
        """
        # Validate order
        if order < 1:
            raise ValueError("Order must be a positive integer.")

        # Validate and apply window
        start, end = window
        if start is None:
            start = 0
        if end is None:
            end = len(self.ccr_matrix)
        
        if start < 0 or end > len(self.ccr_matrix) or start >= end:
            raise ValueError("Invalid window indices.")

        # Slice the data according to the window
        data = self.ccr_matrix.iloc[start:end]

        # Fit and store the VAR model
        model = sm.VAR(data).fit(order)
        self.model = model

        return model

    def var_forecast(self, steps: int = 1) -> pd.DataFrame:
        """
        Generate forecasts using the fitted VAR model.

        Parameters:
        - steps (int): Number of steps ahead to forecast. Default is 1.

        Returns:
        - pd.DataFrame: Forecasted values for each stock.

        Raises:
        - ValueError: If the model hasn't been fitted yet.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_var() first.")

        # Generate forecast
        forecast = self.model.forecast(self.ccr_matrix.values[-self.model.k_ar:], steps=steps)

        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast, columns=self.ccr_matrix.columns)
        forecast_df.index = pd.date_range(start=self.ccr_matrix.index[-1] + pd.Timedelta(days=1), periods=steps)

        return forecast_df

    
class GARCHModel(DataPreprocessor):
    """
    Handle GARCH model operations.
    """
    def __init__(self, tickers: Union[str, List[str], np.array], stock: str = None, period: str = "ytd", interval: str = "1d") -> None:
        super().__init__(tickers, period, interval)
        
        if stock:
            if isinstance(tickers, (np.array, list)):
                if stock not in self.tickers:
                    raise ValueError(f"Stock {stock} not in the list of tickers.")
                self.stock = stock
                self.ccr = self.ccr_matrix[stock]
            elif isinstance(tickers, str):
                if stock != tickers:
                    raise ValueError(f"Specified stock {stock} does not match the single ticker {tickers}")
                self.stock = stock
                self.ccr = self.ccr_matrix[stock]
        else:
            if isinstance(tickers, str):
                self.stock = tickers
                self.ccr = self.ccr_matrix[tickers]
            elif len(tickers) == 1:
                self.stock = tickers[0]
                self.ccr = self.ccr_matrix[tickers[0]]
            else:
                raise ValueError("Must specify a stock when multiple tickers are provided")
        
        self.resids = None
        self.egarch = None

    def garch_order(self, max_p):
        """
        Perform stepwise GARCH order selection using AIC.

        This method iteratively fits GARCH models with different orders up to the specified
        maximum p and q values, selecting the best order based on the Akaike Information 
        Criterion (AIC). It also checks for residual autocorrelation using the Ljung-Box test.

        Parameters:
        - max_p (int): The maximum number of GARCH (p) terms to consider.

        Returns:
        - best_order (Tuple[int, int]): A tuple containing the best GARCH order (p, q) as determined by AIC.

        Raises:
        - ValueError: If max_p is less than 1.
        - RuntimeError: If no valid GARCH model is found within the specified order range.

        Warning: 
        - This method can be computationally intensive for large max_p values or long time series.
        """
        best_order = (0, 0)

        max_q = 1
        aic1, aic2 = np.inf, np.inf - 1

        # Determine the maximum feasible ARCH terms to add
        while aic2 < aic1:
            aic1 = aic2
            # Fit ARCH model
            arch_model = arch.arch_model(self.ccr, vol='Garch', p=0, o=0, q=max_q).fit(disp='off')
            # Calculate AIC
            aic2 = arch_model.aic
            # Increment parameters
            max_q += 1
        max_q -= 1

        lowest_aic = np.inf

        # Iterate through possible combinations of p and q
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    # Fit GARCH model
                    model = arch.arch_model(self.returns_series, vol='Garch', p=p, o=0, q=q).fit(disp='off')

                    # Check residual autocorrelation using Ljung-Box test
                    std_resids = model.resid / model.conditional_volatility
                    lb_test = acorr_ljungbox(std_resids, lags=min(10, len(model.resid)-1), return_df=True)
                    if lb_test[1][0] < 0.05:  # Significance Level
                        continue  # Skip this order if autocorrelation is significant

                    # Update best order if current AIC is lower
                    if model.aic < lowest_aic:
                        lowest_aic = model.aic
                        best_order = p, q

                except Exception as e:
                    print(f"Error fitting GARCH({p}, 0, {q}): {e}")
                    continue
        
        return best_order
    
    def fit_egarch(self, order: tuple[int,int] = None, max_p: int = 10) -> arch.arch_model:
        """
        Fit an EGARCH model to the time series data.

        If no order is provided, this method identifies the best-fitting EGARCH model
        for the time series data stored in `self.ccr` using stepwise selection
        that minimizes the Akaike Information Criterion (AIC).

        Parameters:
        - order (Tuple[int, int], optional): The order of the EGARCH model (p, q). If None, the best order will be determined
            using stepwise selection up to max_p. Default is None.
        - max_p (int, optional): The maximum number of ARCH terms to consider in stepwise selection if order is None.
            Default is 10.

        Returns:
        - The fitted EGARCH model (arch.ARCHModelResult).

        Raises:
        - ValueError: If the provided order is invalid or if max_p is less than 1.
        """
        # Define parameters for model order
        if not order:
            order = self.garch_order(max_p)

        # Fit the model
        model = arch.arch_model(self.ccr, vol='EGARCH', p=order[0], q=order[1], dist='t').fit()
        
        # Compute and store the standardized residuals
        self.resids = model.resid / model.conditional_volatility
        
        # Return the model corresponding to the given window and order
        self.egarch = model
        return model


class ModelEvaluator(ARMAModel, GARCHModel):
    """
    Evaluate the performance of predictive models.
    """
    def rolling_win_sharpe(self, arma_order: tuple[int, int, int], win_size: int = 100, refit_freq: int = 5, c: float = 0.8) -> float:
        """
        Calculate the mean profit from the given model using a rolling window approach.  Refit Periodically.

        Parameters:
        - arma_order (tuple): Order of the model being used.
        - win_size (int): Size of the rolling window. Default is 100.
        - refit_freq (int): Frequency of refitting the model. Default is 5.
        - c (float): Threshold for buying and selling

        Returns:
        - Sharpe Ratio attained using this model and threshold (float).
        """
        forecasts = np.zeros(len(self.ccr) - win_size)
        oos_returns = np.zeros(len(self.ccr) - win_size)

        # Create the forecast array
        for i in range(win_size, len(self.ccr)):
            
            # Define an iterable that corresponds to the index in the forecast array
            forecast_idx = i - win_size
            
            if forecast_idx % refit_freq == 0:
                
                # Refit the model
                window = forecast_idx, i
                model = self.fit_arma(arma_order, window)
            
            # Make one-step ahead forecast
            forecast = self.arma_forecast(model, steps_ahead=1)[0]
        
            # Store the forecast
            forecasts[forecast_idx] = forecast

            # Calculate out-of-sample return
            if forecast > c:
                oos_returns[forecast_idx] = self.ccr.iloc[i]
            elif forecast < -c:
                oos_returns[forecast_idx] = -self.ccr.iloc[i]
            else:
                oos_returns[forecast_idx] = 0
                
            # Update the model with the actual value (without refitting parameters)
            if hasattr(model, 'update'):
                model.update(self.ccr.iloc[i])

        # Calculate the average out-of-sample profit
        pi_bar = np.mean(oos_returns)
        
        # Calculate the out-of-sample standard deviation
        s = np.std(oos_returns)

        # Calculate and return the Sharpe Ratio
        if s == 0:
            return 0
        return pi_bar / s
    
    
class RiskMetrics(ModelEvaluator):
    """
    Calculate various risk metrics.
    """
    def __init__(self, ticker: str, period: str = "ytd", interval: str = "1d", garch=None):
        super().__init__(ticker, period, interval)
        if self.egarch == None or self.resids == None:  # Ensure that the EGARCH model exists
            raise ValueError("EGARCH model not fitted.  Call fit_egarch to attain model and residuals")
    
    def sim_returns(self, n_sims: int, horizon: int) -> np.array:
        """
        Simulate future returns using the fitted GARCH model and bootstrapped residuals.

        This method generates Monte Carlo simulations of future returns based on the
        previously fitted GARCH model. It uses bootstrapping of standardized residuals
        to capture non-normality and other empirical characteristics of the returns
        distribution.

        Parameters:
        - n_sims (int): The number of simulation paths to generate.
        - horizon (int): The number of future time steps to simulate for each path.

        Returns:
        - A 2D numpy array of shape (n_sims, horizon) containing the simulated returns. Each 
          row represents a single simulation path, and each column represents a time step in 
          the future (np.array)

        Raises:
        - ValueError: If the residuals are not available, indicating that the GARCH model
          has not been properly fitted.
        """
        try:
            simulated_returns = np.zeros((n_sims, horizon))
            for i in range(n_sims):
                boot_residuals = np.random.choice(self.resids, size=horizon, replace=True)
                sim = self.garch.simulate(horizon=horizon, initial_value=self.ccr.iloc[-1], custom_dist=boot_residuals)
                simulated_returns[i] = sim.values[:,0]
            return simulated_returns
        except Exception as e:
            print(f"Error in simulating returns: {str(e)}")
            raise

    def calculate_hybrid_var(self, conf_lvl: float = 0.95, horizon: int = 1, n_sims: int = 10000) -> float:
        """
        Calculate Value at Risk using a hybrid approach combining GARCH with t-distributed innovations and bootstrapping.

        Parameters:
        - confidence_level (float): The confidence level for VaR calculation. Default is 0.95.
        - horizon (int): The time horizon for VaR in days. Default is 1.
        - n_simulations (int): The number of Monte Carlo simulations. Default is 10000.

        Returns:
        - The calculated VaR value (float).

        This method uses a GARCH model with t-distributed errors to capture volatility clustering and fat tails,
        then applies bootstrapping to the standardized residuals to account for any remaining non-normality.
        """
        if not 0 < conf_lvl < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("Horizon and number of simulations must be positive integers")
        
        simulated_returns = self.sim_returns(n_sims, horizon)
        return -np.percentile(simulated_returns.sum(axis=1), 100 * (1 - conf_lvl))

