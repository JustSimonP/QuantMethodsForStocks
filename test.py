# -*- coding: utf-8 -*-


        # Merge into one sequence
        if close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
        else:
            self.raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = numpy.array(self.raw_seq)
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)
