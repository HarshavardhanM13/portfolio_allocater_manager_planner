
async function analyze() {
    const portfolio = JSON.parse(
        document.getElementById("portfolio").value
    );

    const user_needs = document.getElementById('user_needs').value;
    console.log(user_needs)
    const res = await fetch("/analyze-portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            user_portfolio: portfolio,
            user_needs, 
        })
    });

    const data = await res.json();

    document.getElementById("evaluation").innerText =
        data.evaluation || "No evaluation generated";

    document.getElementById("allocation").innerText =
        data.allocation_plan || "No allocation plan generated";
}
