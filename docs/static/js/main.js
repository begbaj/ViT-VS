$(document).ready(function() {
  // Tabs functionality
  $('.tabs li').on('click', function() {
    // Remove active class from all tabs
    $('.tabs li').removeClass('is-active');
    $('.tab-content').removeClass('is-active');
    
    // Add active class to clicked tab
    $(this).addClass('is-active');
    
    // Show corresponding content
    const targetId = $(this).data('target');
    $('#' + targetId).addClass('is-active');
    
    // Pause all videos when switching tabs
    $('video').each(function() {
      this.pause();
    });
  });
  
  // Citation modal
  $('#cite-button').on('click', function(e) {
    e.preventDefault();
    $('#citation-modal').addClass('is-active');
  });
  
  // Close citation modal
  $('.modal-background, .modal-card-head .delete').on('click', function() {
    $('#citation-modal').removeClass('is-active');
  });
  
  // Copy citation to clipboard
  $('#copy-citation').on('click', function() {
    const citationText = $('#bibtex-citation').text();
    
    // Create a temporary textarea element to copy from
    const $temp = $('<textarea>');
    $('body').append($temp);
    $temp.val(citationText).select();
    
    // Execute copy command
    document.execCommand('copy');
    
    // Remove the temporary textarea
    $temp.remove();
    
    // Change button text temporarily
    const $button = $(this);
    const originalText = $button.html();
    $button.html('<span class="icon"><i class="fas fa-check"></i></span><span>Copied!</span>');
    
    // Reset button text after 2 seconds
    setTimeout(function() {
      $button.html(originalText);
    }, 2000);
  });
  
  // Handle video loading
  $('video').each(function() {
    // Add loading class
    $(this).on('loadstart', function() {
      $(this).closest('.video-container').addClass('is-loading');
    });
    
    // Remove loading class when video can play
    $(this).on('canplay', function() {
      $(this).closest('.video-container').removeClass('is-loading');
    });
  });

  // Lazy load videos when tab is clicked
  $('.tabs li').on('click', function() {
    const targetId = $(this).data('target');
    const video = $('#' + targetId + ' video')[0];
    
    if (video && !video.getAttribute('src')) {
      const source = video.querySelector('source');
      if (source) {
        video.setAttribute('src', source.getAttribute('src'));
        video.load();
      }
    }
  });
});
