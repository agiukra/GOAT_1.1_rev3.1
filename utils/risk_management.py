def calculate_position_size(balance, risk_percent, stop_loss_distance):
    risk_amount = balance * risk_percent
    return risk_amount / stop_loss_distance

def apply_risk_management(balance, entry_price, stop_loss_price):
    stop_loss_distance = entry_price - stop_loss_price
    return calculate_position_size(balance, 0.01, stop_loss_distance)
