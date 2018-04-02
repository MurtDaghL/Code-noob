import pygame
from time import time, sleep

# Paramètres:
largeur_affichage = 1080
hauteur_affichage = 720
fps = 60

noir = (0,0,0)
blanc = (255,255,255)
rouge = (255,0,0)

vitesse = 500/fps


# Création de la fenêtre:
pygame.init()
affichage = pygame.display.set_mode((largeur_affichage,hauteur_affichage))
pygame.display.set_caption("Le jeu du futur")
horloge = pygame.time.Clock()

# Création du happy eyeball:
happy_image = pygame.image.load('happy-eyeball.png')
happy_largeur,happy_hauteur = happy_image.get_rect().size
def happy(x,y): affichage.blit(happy_image,(x,y))



def obj_texte(texte,police):
    surface_Texte = police.render(texte, True, rouge)
    return surface_Texte, surface_Texte.get_rect()

def message_centré(texte):
    police = pygame.font.Font('freesansbold.ttf',60)
    surface_Texte, rectangle_Texte = obj_texte(texte, police)
    rectangle_Texte.center = (largeur_affichage*0.5,hauteur_affichage*0.2)
    affichage.blit(surface_Texte, rectangle_Texte)
    pygame.display.update()

def bordure():
    message_centré("C'est interdit par la constitution")
    sleep(2)
    game_loop()





def game_loop():
    x = largeur_affichage * 0.5
    y = hauteur_affichage * 0.5
    fin = False
    
    while not fin:
        start = time()
        # Déplacement:
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_DOWN]: y += vitesse
        if pressed[pygame.K_UP]: y -= vitesse
        if pressed[pygame.K_LEFT]: x -= vitesse
        if pressed[pygame.K_RIGHT]: x += vitesse

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Gestion des bordures:
        if x >largeur_affichage - happy_largeur or x<0:
            bordure()
        if y >hauteur_affichage - happy_hauteur or y<0:
            bordure()

        # Raffraîchissemnt:
        affichage.fill(blanc)
        happy(x,y)
        pygame.display.update()
        horloge.tick(fps)

        # Affichage des fps:
        end = time()
        print(1/(end-start))


game_loop()

