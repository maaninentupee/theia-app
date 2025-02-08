    <footer class="site-footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-info">
                    <p class="copyright">
                        &copy; <?php echo date('Y'); ?> <?php echo get_theme_mod('footer_copyright', 'Magic Jim'); ?>. 
                        <?php echo get_theme_mod('footer_text', 'All rights reserved.'); ?>
                    </p>
                </div>
                <nav class="footer-navigation">
                    <?php
                    wp_nav_menu(array(
                        'theme_location' => 'footer-menu',
                        'container' => false,
                        'menu_class' => 'footer-menu',
                        'fallback_cb' => false
                    ));
                    ?>
                </nav>
                <div class="social-links">
                    <?php if (get_theme_mod('social_facebook')): ?>
                        <a href="<?php echo esc_url(get_theme_mod('social_facebook')); ?>" target="_blank">Facebook</a>
                    <?php endif; ?>
                    <?php if (get_theme_mod('social_twitter')): ?>
                        <a href="<?php echo esc_url(get_theme_mod('social_twitter')); ?>" target="_blank">Twitter</a>
                    <?php endif; ?>
                    <?php if (get_theme_mod('social_instagram')): ?>
                        <a href="<?php echo esc_url(get_theme_mod('social_instagram')); ?>" target="_blank">Instagram</a>
                    <?php endif; ?>
                </div>
            </div>
        </div>
    </footer>

    <?php wp_footer(); ?>
</body>
</html>
