/* global bootstrap: false */
(() => {
  'use strict'
  const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  tooltipTriggerList.forEach(tooltipTriggerEl => {
    new bootstrap.Tooltip(tooltipTriggerEl)
  })
})()

/**
 * Easy selector helper function
 */
const select = (el, all = false) => {
  el = el.trim()
  if (all) {
    return [...document.querySelectorAll(el)]
  } else {
    return document.querySelector(el)
  }
}

/**
 * Easy event listener function
 */
const on = (type, el, listener, all = false) => {
  if (all) {
    select(el, all).forEach(e => e.addEventListener(type, listener))
  } else {
    select(el, all).addEventListener(type, listener)
  }
}

/**
 * Easy on scroll event listener 
 */
const onscroll = (el, listener) => {
  el.addEventListener('scroll', listener)
}

/**
 * Sidebar toggle
 */
if (select('.toggle-sidebar-btn')) {
  on('click', '.toggle-sidebar-btn', function(e) {
    select('body').classList.toggle('toggle-sidebar')
  })
}


document.addEventListener("DOMContentLoaded", function() {
  const dataTable = new simpleDatatables.DataTable("#myTable", {
    sortable: true, // Enable sorting
    searchable: true, // Enable searching
    perPage: 10, // Number of rows per page
  });
});

window.onload = function () {
  function showSpinner() {
      document.getElementById("loading").classList.remove("d-none");
      document.getElementById("submit-button").setAttribute("disabled", "true");
  }

  // Add an event listener to the form to trigger the spinner when submitted
  document.getElementById("cv-form").addEventListener("submit", function (event) {
      showSpinner();
      // Optionally, you can prevent the form from submitting immediately
      // event.preventDefault();
  });
};