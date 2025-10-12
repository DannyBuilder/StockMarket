//Refreshes the portfolio every 30 seconds
setInterval(updatePortfolio, 30000);

//Update portfolio on page load
document.addEventListener('DOMContentLoaded', function () {
    updatePortfolio();
});

//Adding a stock
document.getElementById('addStockForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const symbol = document.getElementById('symbol').value.toUpperCase();
    const shares = parseInt(document.getElementById('shares').value);

    try {
        const response = await fetch('/add_stock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, shares })
        });

        const result = await response.json();

        if (result.success) {
            showAlert('success', result.success);
            document.getElementById('addStockForm').reset();
            updatePortfolio();
        } else {
            showAlert('error', result.error);
        }

        updatePortfolio();
    } catch (err) {
        showAlert('error', 'Error adding stock: ' + err.message);
    }
});

// Updates portfolio summary and visuals
async function updatePortfolio() {
    try {
        const response = await fetch('/portfolio_summary');
        const summary = await response.json();

        document.getElementById('totalValue').textContent = '$' + summary.total_value.toLocaleString();
        document.getElementById('cash').textContent = '$' + summary.cash.toLocaleString();

        const gainLoss = document.getElementById('gainLoss');
        gainLoss.textContent = '$' + summary.total_gain_loss.toLocaleString();
        gainLoss.className = summary.total_gain_loss >= 0 ? 'stat-value positive' : 'stat-value negative';

        const gainPercent = document.getElementById('gainLossPercent');
        gainPercent.textContent = summary.total_gain_loss_percent.toFixed(2) + '%';
        gainPercent.className = summary.total_gain_loss_percent >= 0 ? 'stat-value positive' : 'stat-value negative';

        updatePositionsTable(summary.positions);
    } catch (err) {
        console.error('Error updating portfolio', err);
    }
}

//Update the position table
function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsBody');
    tbody.innerHTML = '';

    positions.forEach(pos => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${pos.symbol}</strong></td>
            <td>${pos.company_name}</td>
            <td>${pos.shares}</td>
            <td>$${pos.avg_price.toFixed(2)}</td>
            <td>$${pos.current_price.toFixed(2)}</td>
            <td>$${pos.market_value.toFixed(2)}</td>
            <td class="${pos.gain_loss >= 0 ? 'positive' : 'negative'}">
                $${pos.gain_loss.toFixed(2)} (${pos.gain_loss_percent.toFixed(2)}%)
            </td>
            <td class="${pos.ytd_percent >= 0 ? 'positive' : 'negative'}">
                ${pos.ytd_percent.toFixed(2)}%
            </td>
            <td>
                <button class="btn btn-danger" onclick="sellStock('${pos.symbol}')" 
                    style="width:auto;padding:5px 10px;font-size:0.8rem;">
                    <i class="fas fa-minus"></i> Sell
                </button>
            </td>`;
        tbody.appendChild(row);
    });

    document.getElementById('positionsTable').style.display = positions.length > 0 ? 'table' : 'none';
}

//Handle selling stocks
async function sellStock(symbol) {
    const shares = prompt(`How many shares of ${symbol} do you want to sell?`);

    if (shares && parseInt(shares) > 0) {
        try {
            const response = await fetch('/sell_stock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, shares: parseInt(shares) })
            });

            const result = await response.json();

            if (result.success) {
                showAlert('success', result.success);
                updatePortfolio();
            } else {
                showAlert('error', result.error);
            }
        } catch (err) {
            showAlert('error', 'Error selling stock: ' + err.message);
        }
    }

    updatePortfolio();
}

//Display errors
function showAlert(type, message) {
    const el = document.getElementById(type === 'success' ? 'successAlert' : 'errorAlert');
    el.textContent = message;
    el.style.display = 'block';

    setTimeout(() => { el.style.display = 'none'; }, 5000);
}
//Hides alerts after 5 seconds
window.onload = function() {
    const successAlert = document.getElementById('successAlert');
    const errorAlert = document.getElementById('errorAlert');   
    if (successAlert) setTimeout(()=>{successAlert.style.display='none'},5000);
    if (errorAlert) setTimeout(()=>{errorAlert.style.display='none'},5000);
}