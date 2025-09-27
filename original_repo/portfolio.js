const portfolio = [];


async function addStock() {
  const ticker = document.getElementById("ticker").value.toUpperCase();
  const shares = parseInt(document.getElementById("shares").value);
  if (!ticker || !shares) return;

  const price = await fetchPrice(ticker);
  portfolio.push({ ticker, shares, price });

  updateTable();
  document.getElementById("ticker").value = "";
  document.getElementById("shares").value = "";
}

function updateTable() {
  const tbody = document.querySelector("#portfolioTable tbody");
  tbody.innerHTML = ""; // Clear table

  let totalValue = 0;

  portfolio.forEach(stock => {
    const row = document.createElement("tr");
    const value = (stock.shares * stock.price).toFixed(2);
    totalValue += parseFloat(value);

    row.innerHTML = `
      <td>${stock.ticker}</td>
      <td>${stock.shares}</td>
      <td>$${stock.price.toFixed(2)}</td>
      <td>$${value}</td>
    `;
    tbody.appendChild(row);
  });

  // Show total portfolio value
  const totalRow = document.createElement("tr");
  totalRow.innerHTML = `<td colspan="3" style="font-weight:bold;">Total Value</td><td style="font-weight:bold;">$${totalValue.toFixed(2)}</td>`;
  tbody.appendChild(totalRow);
}


document.getElementById("addStockBtn").addEventListener("click", addStock);
