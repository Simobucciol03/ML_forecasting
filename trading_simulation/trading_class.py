import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TradingSimulation:
    def __init__(self, df, model_predictions, initial_balance=100000, risk_per_trade=0.01, stop_loss_pct=0.02, take_profit_pct=0.05, commission_pct=0.001):
        """
        Inizializza la simulazione di trading.

        :param df: DataFrame con i dati storici (prezzi storici)
        :param model_predictions: Predizioni del modello (0 per short, 1 per long)
        :param initial_balance: Capitale iniziale per la simulazione
        :param risk_per_trade: Percentuale del capitale da rischiare per ogni operazione
        :param stop_loss_pct: Percentuale per lo stop loss (es. 0.02 per il 2%)
        :param take_profit_pct: Percentuale per il take profit (es. 0.05 per il 5%)
        :param commission_pct: Commissione sulle operazioni (es. 0.001 per 0.1%)
        """
        self.df = df
        self.model_predictions = model_predictions
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.commission_pct = commission_pct
        self.positions = []  # Tiene traccia delle posizioni aperte
        self.trade_history = []  # Storia delle transazioni
        self.capital_history = [initial_balance]  # Storico del capitale

    def execute_trade(self, date, position, price):
        """
        Esegue una singola transazione (compra/vendi) in base alla previsione.

        :param date: La data della previsione
        :param position: La posizione prevista (0 = Short, 1 = Long)
        :param price: Il prezzo di mercato
        """
        # Calcolo del rischio per ogni trade
        risk_amount = self.balance * self.risk_per_trade
        trade_size = risk_amount / price  # Quantità da comprare/vendere in base al rischio

        # Applicazione commissione
        commission = price * trade_size * self.commission_pct
        net_price = price + commission  # Prezzo netto con commissione

        if position == 1:  # Long
            self.balance -= net_price * trade_size  # Spende il capitale per comprare
            self.positions.append((date, "Long", trade_size, price, net_price))  # Aggiungi alla lista delle posizioni aperte
            self.trade_history.append((date, "Buy", trade_size, price, commission))
        elif position == 0:  # Short
            self.balance += net_price * trade_size  # Aggiungi il capitale dalla vendita
            self.positions.append((date, "Short", trade_size, price, net_price))
            self.trade_history.append((date, "Sell", trade_size, price, commission))

    def close_position(self, date, position, close_price):
        """
        Chiude una posizione aperta con il prezzo di chiusura.
        
        :param date: La data della chiusura della posizione
        :param position: La posizione (Long o Short)
        :param close_price: Il prezzo di chiusura
        """
        for i, pos in enumerate(self.positions):
            entry_price = pos[3]
            trade_size = pos[2]
            net_price = pos[4]
            stop_loss_price = entry_price * (1 - self.stop_loss_pct) if position == 1 else entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct) if position == 1 else entry_price * (1 - self.take_profit_pct)

            if (position == 1 and (close_price <= stop_loss_price or close_price >= take_profit_price)):
                self.balance += trade_size * (close_price - entry_price)  # Guadagno dalla posizione long
                self.trade_history.append((date, "Sell", trade_size, close_price, 0))  # Vendi la posizione long
                del self.positions[i]
                break
            elif (position == 0 and (close_price >= stop_loss_price or close_price <= take_profit_price)):
                self.balance += trade_size * (entry_price - close_price)  # Guadagno dalla posizione short
                self.trade_history.append((date, "Buy", trade_size, close_price, 0))  # Riacquista la posizione short
                del self.positions[i]
                break

    def simulate_trading(self):
        """
        Esegue la simulazione di trading su base giornaliera.
        """
        for i in range(1, len(self.df)):
            date = self.df.index[i]
            price = self.df['close'].iloc[i]
            prediction = self.model_predictions[i]

            # Se la previsione è Long, esegui l'acquisto se non hai già una posizione
            if prediction == 1 and not any(pos[1] == "Long" for pos in self.positions):
                self.execute_trade(date, 1, price)

            # Se la previsione è Short, esegui la vendita se non hai già una posizione
            elif prediction == 0 and not any(pos[1] == "Short" for pos in self.positions):
                self.execute_trade(date, 0, price)

            # Chiudi le posizioni aperte
            for pos in self.positions:
                if pos[1] == "Long":
                    self.close_position(date, 1, price)
                elif pos[1] == "Short":
                    self.close_position(date, 0, price)

            # Aggiungi il saldo corrente nella cronologia
            self.capital_history.append(self.balance)

    def plot_performance(self):
        """
        Visualizza le performance del trading con segnali di acquisto e vendita.
        """
        plt.plot(self.df.index, self.capital_history, label="Capitale durante il trading", color='blue')
        buy_dates = [trade[0] for trade in self.trade_history if trade[1] == "Buy"]
        buy_prices = [trade[3] for trade in self.trade_history if trade[1] == "Buy"]
        sell_dates = [trade[0] for trade in self.trade_history if trade[1] == "Sell"]
        sell_prices = [trade[3] for trade in self.trade_history if trade[1] == "Sell"]
        
        plt.scatter(buy_dates, buy_prices, marker="^", color="green", label="Compra")
        plt.scatter(sell_dates, sell_prices, marker="v", color="red", label="Vendi")
        
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.title('Trading Simulation Performance with Buy/Sell Signals')
        plt.legend()
        plt.show()

    def display_trade_history(self):
        """
        Mostra la cronologia delle transazioni.
        """
        print("Trading History:")
        for trade in self.trade_history:
            print(f"Date: {trade[0]}, Action: {trade[1]}, Size: {trade[2]}, Price: {trade[3]:.2f}, Commission: {trade[4]:.4f}")


