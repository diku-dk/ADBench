module ba = import "ba"
module gmm = import "gmm"
module hand = import "hand"

entry ba_calculate_objective = ba.calculate_objective
entry ba_calculate_jacobian = ba.calculate_jacobian

entry gmm_calculate_objective = gmm.calculate_objective
entry gmm_calculate_jacobian = gmm.calculate_jacobian

entry hand_calculate_objective = hand.calculate_objective
entry hand_calculate_jacobian = hand.calculate_jacobian
