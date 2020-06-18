import math
from six.moves import xrange
from catboost import Pool, CatBoostClassifier


class FocalLossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        gamma = 2.
        # alpha = 1.
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        
        exponents = []
        for index in xrange(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in xrange(len(targets)):
            p = exponents[index] / (1 + exponents[index])

            if targets[index] > 0.0:
                der1 = -((1-p)**(gamma-1))*(gamma * math.log(p) * p + p - 1)/p
                der2 = gamma*((1-p)**gamma)*((gamma*p-1)*math.log(p)+2*(p-1))
            else:
                der1 = (p**(gamma-1)) * (gamma * math.log(1 - p) - p)/(1 - p)
                der2 = p**(gamma-2)*((p*(2*gamma*(p-1)-p))/(p-1)**2 + (gamma-1)*gamma*math.log(1 - p))

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result



if __name__ == '__main__':
    cat_features = [0, 1]

    train_data = [[1, 4, 5, 6],
                [4, 5, 6, 7],
                [30, 40, 50, 60]]

    train_labels = [1, 1, 0]

    eval_data = [[3, 4, 4, 1],
                [1, 5, 5, 5],
                [31, 25, 60, 70]]

    # Initialize CatBoostClassifier with custom `loss_function`
    model = CatBoostClassifier(loss_function=FocalLossObjective(),
                            eval_metric="Logloss")
    # Fit model
    model.fit(train_data, train_labels)
    # Only prediction_type='RawFormulVal' allowed with custom `loss_function`
    preds_raw = model.predict(eval_data,
                            prediction_type='RawFormulaVal')

    print(preds_raw)