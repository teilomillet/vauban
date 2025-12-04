import unittest
from vauban.strategies.gepa.map_elites import MAPElitesArchive
from vauban.interfaces import AttackPrompt

class TestMAPElitesArchive(unittest.TestCase):
    def test_add_and_replace(self):
        archive = MAPElitesArchive()
        
        # Create dummy attacks
        attack1 = AttackPrompt(prompt="test1", strategy="StyleA")
        candidate1 = {
            "attack": attack1,
            "score": 5.0,
            "stealth": 2.0, # Low stealth bucket
        }
        
        # Add first candidate
        added = archive.add(candidate1)
        self.assertTrue(added)
        self.assertEqual(len(archive.get_all()), 1)
        
        # Try adding a worse candidate to the same niche
        attack2 = AttackPrompt(prompt="test2", strategy="StyleA")
        candidate2 = {
            "attack": attack2,
            "score": 3.0,
            "stealth": 2.0, # Same niche
        }
        added = archive.add(candidate2)
        self.assertFalse(added)
        self.assertEqual(len(archive.get_all()), 1)
        self.assertEqual(archive.get_all()[0]["score"], 5.0)
        
        # Add a better candidate to the same niche
        attack3 = AttackPrompt(prompt="test3", strategy="StyleA")
        candidate3 = {
            "attack": attack3,
            "score": 7.0,
            "stealth": 2.0, # Same niche
        }
        added = archive.add(candidate3)
        self.assertTrue(added)
        self.assertEqual(len(archive.get_all()), 1)
        self.assertEqual(archive.get_all()[0]["score"], 7.0)
        
        # Add candidate to a different niche (different stealth)
        attack4 = AttackPrompt(prompt="test4", strategy="StyleA")
        candidate4 = {
            "attack": attack4,
            "score": 4.0,
            "stealth": 9.0, # High stealth bucket
        }
        added = archive.add(candidate4)
        self.assertTrue(added)
        self.assertEqual(len(archive.get_all()), 2)
        
        # Add candidate to a different niche (different style)
        attack5 = AttackPrompt(prompt="test5", strategy="StyleB")
        candidate5 = {
            "attack": attack5,
            "score": 6.0,
            "stealth": 2.0, # Low stealth bucket
        }
        added = archive.add(candidate5)
        self.assertTrue(added)
        self.assertEqual(len(archive.get_all()), 3)

    def test_select_survivors(self):
        archive = MAPElitesArchive()
        
        # Populate archive with 10 items
        for i in range(10):
            attack = AttackPrompt(prompt=f"test{i}", strategy=f"Style{i}")
            candidate = {
                "attack": attack,
                "score": float(i),
                "stealth": 5.0,
            }
            archive.add(candidate)
            
        survivors = archive.select_survivors(5)
        self.assertEqual(len(survivors), 5)
        
        survivors_all = archive.select_survivors(20)
        self.assertEqual(len(survivors_all), 10)

if __name__ == '__main__':
    with open("test_results.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)

