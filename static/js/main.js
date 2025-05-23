document.addEventListener('DOMContentLoaded', function() {
    // Animate elements when they come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Get delay attribute if it exists
                const delay = entry.target.getAttribute('data-delay') || 0;
                
                // Apply animation with delay
                setTimeout(() => {
                    entry.target.classList.add('animated');
                    
                    // For elements with animate__animated class but no specific animation
                    if (entry.target.classList.contains('animate__animated') && 
                        !Array.from(entry.target.classList).some(cls => cls.startsWith('animate__') && cls !== 'animate__animated')) {
                        entry.target.classList.add('animate__fadeIn');
                    }
                    
                    // For animate-list-item elements
                    if (entry.target.classList.contains('animate-list-item')) {
                        entry.target.style.opacity = 1;
                        entry.target.style.transform = 'translateX(0)';
                    }
                    
                    // For feature cards
                    if (entry.target.classList.contains('feature-card')) {
                        entry.target.style.opacity = 1;
                        entry.target.style.transform = 'translateY(0)';
                    }
                }, delay);
            }
        });
    }, {
        threshold: 0.1
    });
    
    // Observe elements with animation classes
    document.querySelectorAll('.animate__animated:not(.animate__infinite), .animate-list-item, .feature-card, .section-title').forEach(element => {
        // For animate-list-item elements
        if (element.classList.contains('animate-list-item')) {
            element.style.opacity = 0;
            element.style.transform = 'translateX(20px)';
            element.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
        }
        
        // For feature cards
        if (element.classList.contains('feature-card')) {
            element.style.opacity = 0;
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        }
        
        // For section titles
        if (element.classList.contains('section-title')) {
            element.classList.add('animate__animated', 'animate__fadeIn');
        }
        
        observer.observe(element);
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animate process steps in sequence
    const processSteps = document.querySelectorAll('.process-step');
    if (processSteps.length > 0) {
        const processObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                processSteps.forEach((step, index) => {
                    setTimeout(() => {
                        step.classList.add('animate__animated', 'animate__fadeInRight');
                        step.style.opacity = 1;
                    }, index * 200);
                });
            }
        }, { threshold: 0.3 });
        
        processObserver.observe(document.querySelector('.process-flow'));
        
        // Set initial opacity
        processSteps.forEach(step => {
            step.style.opacity = 0;
        });
    }
    
    // Animate contact items with staggered delay
    const contactItems = document.querySelectorAll('.contact-item');
    if (contactItems.length > 0) {
        contactItems.forEach((item, index) => {
            item.style.opacity = 0;
            item.style.transform = 'translateY(20px)';
            item.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            
            const contactObserver = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) {
                    setTimeout(() => {
                        item.style.opacity = 1;
                        item.style.transform = 'translateY(0)';
                    }, index * 200);
                }
            }, { threshold: 0.3 });
            
            contactObserver.observe(item);
        });
    }
});
