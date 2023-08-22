/* global bootstrap: false */
(() => {
  'use strict'
  const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  tooltipTriggerList.forEach(tooltipTriggerEl => {
    new bootstrap.Tooltip(tooltipTriggerEl)
  })
})()


document.addEventListener("DOMContentLoaded", function() {
  const dataTable = new simpleDatatables.DataTable("#myTable", {
    sortable: true, // Enable sorting
    searchable: true, // Enable searching
    perPage: 10, // Number of rows per page
  });
});
