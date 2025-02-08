<?php get_header(); ?>

<main id="main" class="site-main">
    <section id="about" class="section-about">
        <div class="container">
            <h2><?php echo get_theme_mod('about_title', 'About Me'); ?></h2>
            <p><?php echo get_theme_mod('about_content', 'I\'m Magic Jim, and I am a passionate individual with a love for magic and creativity.'); ?></p>
        </div>
    </section>

    <section id="portfolio" class="section-portfolio">
        <div class="container">
            <h2><?php echo get_theme_mod('portfolio_title', 'My Work'); ?></h2>
            <p><?php echo get_theme_mod('portfolio_content', 'Check out some of my amazing projects!'); ?></p>
            <?php
            // Add portfolio items loop here later
            ?>
        </div>
    </section>

    <section id="contact" class="section-contact">
        <div class="container">
            <h2><?php echo get_theme_mod('contact_title', 'Contact Me'); ?></h2>
            <p><?php echo get_theme_mod('contact_content', 'Feel free to reach out!'); ?></p>
            <?php
            // Add contact form here later
            ?>
        </div>
    </section>
</main>

<?php get_footer(); ?>
