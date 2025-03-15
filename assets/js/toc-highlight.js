document.addEventListener('DOMContentLoaded', function() {
  // Get all headings in the content (including h1)
  const headings = document.querySelectorAll('.main-content h1, .main-content h2, .main-content h3, .main-content h4');
  const tocLinks = document.querySelectorAll('.toc-container a');

  if (headings.length === 0 || tocLinks.length === 0) return;

  // Create an array of heading positions
  const headingPositions = [];
  headings.forEach(heading => {
      if (heading.id) { // Ensure the heading has an ID
          headingPositions.push({
              id: heading.id,
              top: heading.getBoundingClientRect().top + window.pageYOffset - 100 // Offset for better UX
          });
      }
  });

  if (headingPositions.length === 0) return; //if there are no headings with IDs, stop.

  // Function to highlight the current section
  function highlightToc() {
    const scrollPosition = window.pageYOffset;
    let currentSection = null;

    for (let i = headingPositions.length - 1; i >= 0; i--) {
        if (scrollPosition >= headingPositions[i].top) {
            currentSection = headingPositions[i].id;
            break; // Found the current section, exit the loop
        }
    }

    // Remove all active classes
    tocLinks.forEach(link => {
        link.classList.remove('active');
    });

    // Add active class to current section link
    if (currentSection) { // Only add active class if a section was found
        const activeLink = document.querySelector(`.toc-container a[href="#${currentSection}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
}

  // Listen for scroll events
  window.addEventListener('scroll', function() {
      requestAnimationFrame(highlightToc);
  });

  // Initialize on page load
  highlightToc();
});