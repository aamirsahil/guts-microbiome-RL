// Add an event listener to each number input field
const numFieldsInputs = document.querySelectorAll(".num-fields-input");
numFieldsInputs.forEach(function(numFieldsInput) {
  numFieldsInput.addEventListener("input", function() {
    // Get the number of fields specified by the user
    const numFields = numFieldsInput.value;
    // Get the ID of the corresponding subcards container
    const subcardsContainerId = numFieldsInput.closest(".card-header").dataset.target;
    const subcardsContainer = document.getElementById(subcardsContainerId);
    // Clear the subcards container
    subcardsContainer.innerHTML = "";
    // Add the specified number of range input fields as subcards
    for (let i = 0; i < numFields; i++) {
      const subcard = document.createElement("div");
      subcard.className = "card subcard";
      subcard.innerHTML = `
        <div class="card-header">
          <h6>Range Field ${i + 1}</h6>
        </div>
        <div class="card-body">
          <input type="range" min="0" max="100" step="1" value="50">
        </div>
      `;
      subcardsContainer.appendChild(subcard);
    }
  });
});