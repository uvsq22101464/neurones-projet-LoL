from reseau import predict
from reseau import showPrediction
from reseau import chart
from reseau import analyze_predictions
from reseau import Network1

pred = predict(network=Network1())
#showPrediction(pred)
#chart(pred["volibear"], "Volibear")
analyze_predictions(pred)