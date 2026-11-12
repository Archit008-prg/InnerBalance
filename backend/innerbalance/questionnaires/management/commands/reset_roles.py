from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from patients.models import Patient, Doctor


class Command(BaseCommand):
    help = 'Wipe all Doctor and Patient profile rows (keeps Django auth_user rows intact).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--users',
            action='store_true',
            help='Also delete the underlying auth_user rows (full reset).',
        )

    def handle(self, *args, **options):
        p_count = Patient.objects.count()
        d_count = Doctor.objects.count()

        Patient.objects.all().delete()
        Doctor.objects.all().delete()

        self.stdout.write(self.style.SUCCESS(
            f'Deleted {p_count} patient profile(s) and {d_count} doctor profile(s).'
        ))

        if options['users']:
            # Remove non-superuser accounts as well
            u_count = User.objects.filter(is_superuser=False).count()
            User.objects.filter(is_superuser=False).delete()
            self.stdout.write(self.style.SUCCESS(
                f'Also deleted {u_count} non-superuser auth_user row(s).'
            ))
